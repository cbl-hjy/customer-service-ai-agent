#!/usr/bin/env python3
"""实验 B：RRF 融合参数扫描（k × dense 权重，第三波 RAG 优化，2026-09-16）。

问题：现役 RRF 纯排名融合（k=60，两路等权），从未做过参数搜索；dense 路质量
高于 BM25（R@1 0.422 vs 0.343），等权可能未充分利用稠密信号。

方法：本地离线扫参——重算 BM25 top-20（jieba 分词，与生产同口径）+ 复用
bge-m3 向量缓存重算 dense top-20，网格搜索加权 RRF：
    score(d) = w_dense/(k + rank_dense) + 1.0/(k + rank_bm25)
    k ∈ {20,40,60,100}，w_dense ∈ {0.8,1.0,1.2,1.5}
零 API 成本（纯本地推理，bge-m3 加载一次）。

决策门（方案定标）：最优组合 R@1 比基线（k=60, w=1.0）提升 ≥2pp 才建议落生产。

用法：python eval/kb_v2/sweep_rrf_weights.py
输出：网格表 + 最优组合 + 与基线一致性校验（复算基线应复现 details 的 rrf_top5）
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

# 网格
K_GRID = (20, 40, 60, 100)
W_DENSE_GRID = (0.8, 1.0, 1.2, 1.5)
RECALL_WIDTH = 20  # 两路召回宽度（与生产 RERANK_POOL 口径一致）


def build_bm25_top20():
    """BM25 top-20 全量重算（不过 SCORE_THRESHOLD——融合需要宽池，与生产宽池口径同）。"""
    from kb_retriever import _tokenize  # 与生产同分词/停用词
    import jieba
    from rank_bm25 import BM25Okapi

    kb = json.load(open(KB_PATH, encoding="utf-8"))
    out = {}  # domain -> (index, entries)
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


def bm25_top20(index_entries, query):
    """返回 [(score, title)] top-20（含 0 分条目截断——宽池纪律）。"""
    index, entries = index_entries
    toks = _tokenize_lite(query)
    if not toks:
        return []
    scores = index.get_scores(toks)
    ranked = sorted(((scores[i], entries[i].get("title", "")) for i in range(len(entries))),
                    key=lambda x: x[0], reverse=True)[:RECALL_WIDTH]
    return [(s, t) for s, t in ranked if s > 0]


def _tokenize_lite(query):
    from kb_retriever import _tokenize
    return _tokenize(query)


def build_dense():
    """复用评估侧 DenseRetriever（向量缓存命中则秒级）。"""
    sys.path.insert(0, HERE)
    from dense_retriever import DenseRetriever
    dr = DenseRetriever(kb_path=KB_PATH, cache_path=os.path.join(HERE, "kb_v2_dense_vectors.npz"))
    dr.build_index()
    return dr


def weighted_rrf(bm25_hits, dense_hits, k, w_dense, top_k=5):
    scores = {}
    for rank, (_, t) in enumerate(bm25_hits):
        scores[t] = scores.get(t, 0.0) + 1.0 / (k + rank + 1)
    for rank, (_, t) in enumerate(dense_hits):
        scores[t] = scores.get(t, 0.0) + w_dense / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_k]]


def eval_ranks(titles, expected):
    r1 = 1.0 if titles and titles[0] in expected else 0.0
    mrr = 0.0
    for i, t in enumerate(titles):
        if t in expected:
            mrr = 1.0 / (i + 1)
            break
    return r1, mrr


def main():
    from kb_retriever import _expand_query  # 与生产同查询扩展

    records = json.load(open(DETAILS, encoding="utf-8"))
    n = len(records)
    print(f"扫参基座：{n} 条金标 | 召回宽度 {RECALL_WIDTH} | "
          f"k 网格 {K_GRID} | w_dense 网格 {W_DENSE_GRID}\n")

    print("构建 BM25 索引（本地秒级）...")
    bm25 = build_bm25_top20()
    print("加载 bge-m3 稠密路（向量缓存命中则跳过编码）...")
    dense = build_dense()

    # 预计算每条 query 的两路召回（含生产同款查询扩展）
    cached = []
    for rec in records:
        q = _expand_query(rec["query"])
        b = bm25_top20(bm25[rec["domain"]], q)
        d = dense.search(rec["domain"], q, top_k=RECALL_WIDTH)  # [(score, title)]
        cached.append((rec, b, d))

    # 一致性校验：复算基线（k=60,w=1.0）应大体复现 details 的 rrf_top5
    match = 0
    for rec, b, d in cached:
        if weighted_rrf(b, d, 60, 1.0) == rec["rrf_top5"]:
            match += 1
    print(f"\n一致性校验：复算基线 top5 与 details rrf_top5 完全一致 {match}/{n}"
          f"（{match / n:.0%}；差异源于 details 生成期的池口径/阈值细节，>70% 即可信扫参）\n")

    # 网格扫描
    results = []
    for k in K_GRID:
        for w in W_DENSE_GRID:
            r1s, mrrs = 0.0, 0.0
            for rec, b, d in cached:
                r1, mrr = eval_ranks(weighted_rrf(b, d, k, w), set(rec["expected"]))
                r1s += r1
                mrrs += mrr
            results.append((k, w, r1s / n, mrrs / n))

    base = next(r for r in results if r[0] == 60 and r[1] == 1.0)
    print(f"{'k':>5} {'w_dense':>8} {'R@1':>7} {'MRR':>7}  Δ_vs_基线")
    for k, w, r1, mrr in sorted(results, key=lambda x: x[2], reverse=True):
        delta = r1 - base[2]
        marker = " ←基线" if (k, w) == (60, 1.0) else (" ★最优" if r1 == max(x[2] for x in results) else "")
        print(f"{k:>5} {w:>8.1f} {r1:>7.3f} {mrr:>7.3f}  {delta:+.3f}{marker}")

    best = max(results, key=lambda x: x[2])
    gain = best[2] - base[2]
    print(f"\n最优组合：k={best[0]}, w_dense={best[1]} → R@1={best[2]:.3f}（基线 {base[2]:.3f}，"
          f"提升 {gain:+.3f}）")
    verdict = "✅ 过决策门（≥+0.02）" if gain >= 0.02 else "❌ 未过决策门（<+0.02，不建议落生产）"
    print(f"决策门：{verdict}")


if __name__ == "__main__":
    main()
