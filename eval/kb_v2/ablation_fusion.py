#!/usr/bin/env python3
"""实验 B 消融：拆解 sweep_rrf_weights 与生产口径的三处分歧，量化各自真实贡献。

背景：sweep 名义最优 k=20,w_dense=1.5 → R@1 0.530（+6.9pp），但其口径与生产/评估
（details.json）有三处分歧，增益无法归因：
  1. sweep 的 BM25 宽池不过 SCORE_THRESHOLD（生产 _bm25_hits 过滤 score<0.30）
  2. sweep 无 _exact_hits 短路（生产 exact 命中直接返回，不进融合）
  3. sweep 的 dense 路误用扩展后 query（生产 hybrid.rank 收 raw_query）

方法：完全复刻生产 retrieve() 检索判定（扩展→exact 短路→融合），
对四组配置消融（阈值开关 × 融合参数），指标口径全组一致：
  - exact 命中条目：检索结果 = exact 条目（生产行为，与融合参数无关）
  - 其余条目：weighted RRF top5
指标：HIT@1（top1∈expected）+ MRR，全部 217 条。

配置：
  C0 生产现状   thr=on  k=60  w=1.0
  C1 仅去阈值    thr=off k=60  w=1.0
  C2 仅调参     thr=on  k=20  w=1.5
  C3 阈值+调参  thr=off k=20  w=1.5

用法：python eval/kb_v2/ablation_fusion.py（零 API 成本，dense 复用 npz 缓存）
"""
import json
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DETAILS = os.path.join(HERE, "eval_retrieval_rerank_details.json")
KB_PATH = os.path.join(HERE, "knowledge_base_v2.json")

from kb_retriever import KBRetriever, _expand_query, _tokenize  # noqa: E402

SCORE_THRESHOLD = 0.30
RECALL_WIDTH = 20

CONFIGS = [
    ("C0 生产现状  thr=on  k=60  w=1.0", True, 60, 1.0),
    ("C1 仅去阈值   thr=off k=60  w=1.0", False, 60, 1.0),
    ("C2 仅调参    thr=on  k=20  w=1.5", True, 20, 1.5),
    ("C3 阈值+调参 thr=off k=20  w=1.5", False, 20, 1.5),
]


def build_bm25_raw():
    """BM25 原始分 top-20（不过阈值），返回 domain -> (index, entries)。"""
    from rank_bm25 import BM25Okapi

    kb = json.load(open(KB_PATH, encoding="utf-8"))
    out = {}
    for domain, entries in kb.items():
        if domain.startswith("_"):
            continue

        def entry_text(e):
            parts = [e.get("category", ""), e.get("title", ""), e.get("content", "")]
            kws = e.get("keywords", [])
            if isinstance(kws, list):
                parts.extend(str(k) for k in kws)
            models = e.get("models", [])
            if isinstance(models, list):
                parts.extend(str(m) for m in models)
            return " ".join(p for p in parts if p)

        tokenized = [_tokenize(entry_text(e)) for e in entries]
        out[domain] = (BM25Okapi(tokenized), entries)
    return out


def bm25_hits_raw(index_entries, query, threshold):
    """[(score, title)] top-20；threshold=True 时过滤 score<0.30（生产口径）。"""
    index, entries = index_entries
    toks = _tokenize(query)
    if not toks:
        return []
    scores = index.get_scores(toks)
    ranked = sorted(((scores[i], entries[i].get("title", "")) for i in range(len(entries))),
                    key=lambda x: x[0], reverse=True)[:RECALL_WIDTH]
    hits = [(s, t) for s, t in ranked if s > 0]
    if threshold:
        hits = [(s, t) for s, t in hits if s >= SCORE_THRESHOLD]
    return hits


def weighted_rrf(bm25_hits, dense_hits, k, w_dense, top_k=5):
    scores = {}
    for rank, (_, t) in enumerate(bm25_hits):
        scores[t] = scores.get(t, 0.0) + 1.0 / (k + rank + 1)
    for rank, (_, t) in enumerate(dense_hits):
        scores[t] = scores.get(t, 0.0) + w_dense / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_k]]


def eval_ranks(titles, expected):
    hit1 = 1.0 if titles and titles[0] in expected else 0.0
    mrr = 0.0
    for i, t in enumerate(titles):
        if t in expected:
            mrr = 1.0 / (i + 1)
            break
    return hit1, mrr


def main() -> None:
    records = json.load(open(DETAILS, encoding="utf-8"))
    n = len(records)

    # 生产同款检索器（exact 预检 / 分词 / 阈值判定与其内部实现完全一致）
    prod = KBRetriever(kb_path=KB_PATH)

    print("构建 BM25 原始分索引（本地秒级）...")
    bm25_raw = build_bm25_raw()

    print("加载 bge-m3 稠密路（npz 缓存命中则跳过编码）...")
    sys.path.insert(0, HERE)
    from dense_retriever import DenseRetriever
    dense = DenseRetriever(kb_path=KB_PATH, cache_path=os.path.join(HERE, "kb_v2_dense_vectors.npz"))
    dense.build_index()

    # 预计算：exact 短路结果 + 两路召回（dense 用原 query——生产口径）
    prepared = []
    n_exact = 0
    for rec in records:
        q_raw, domain = rec["query"], rec["domain"]
        expected = set(rec["expected"])
        expanded = _expand_query(q_raw)
        exact = prod._exact_hits(domain, expanded)
        if exact:
            n_exact += 1
            prepared.append((rec, expected, [str(e.get("title") or "") for e in exact], None, None))
            continue
        b_raw = bm25_hits_raw(bm25_raw[domain], expanded, threshold=False)
        d_hits = dense.search(domain, q_raw, top_k=RECALL_WIDTH)
        prepared.append((rec, expected, None, b_raw, d_hits))
    print(f"基座：{n} 条金标 | exact 短路 {n_exact} 条 | 融合路 {n - n_exact} 条\n")

    print(f"{'配置':<34}{'HIT@1':>8}{'MRR':>8}  Δ_HIT@1_vs_C0")
    base_hit = None
    out = []
    for name, thr, k, w in CONFIGS:
        h1s, mrrs = 0.0, 0.0
        for rec, expected, exact_titles, b_raw, d_hits in prepared:
            if exact_titles is not None:
                titles = exact_titles
            else:
                b = b_raw if not thr else [(s, t) for s, t in b_raw if s >= SCORE_THRESHOLD]
                titles = weighted_rrf(b, d_hits, k, w)
            h1, mrr = eval_ranks(titles, expected)
            h1s += h1
            mrrs += mrr
        hit1, mrr = h1s / n, mrrs / n
        if base_hit is None:
            base_hit = hit1
        delta = hit1 - base_hit
        out.append((name, hit1, mrr, delta))
        print(f"{name:<34}{hit1:>8.3f}{mrr:>8.3f}  {delta:+.3f}")

    c1 = next(o for o in out if o[0].startswith("C1"))
    c2 = next(o for o in out if o[0].startswith("C2"))
    c3 = next(o for o in out if o[0].startswith("C3"))
    print(f"\n归因拆解（Δ vs C0 生产现状）：")
    print(f"  去 SCORE_THRESHOLD 贡献：{c1[3]:+.3f}")
    print(f"  融合参数(k=20,w=1.5)贡献：{c2[3]:+.3f}")
    print(f"  两者叠加：{c3[3]:+.3f}")
    gate = "✅ 过决策门（≥+0.02）" if c3[3] >= 0.02 else "❌ 未过决策门"
    print(f"决策门（叠加组合 ≥+0.02 才建议落生产）：{gate}")


if __name__ == "__main__":
    main()
