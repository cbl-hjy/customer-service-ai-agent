#!/usr/bin/env python3
"""fp16 A/B 漂移实测（GPU 加速落地前置实验，2026-09-16，用户已认可验证路径）。

背景（官方文档查证，详见 ai-docs/实验报告）：
  - BAAI 官方背书 bge-reranker-v2-m3 use_fp16（"slight performance degradation"）
  - 但 ST 5.7.0（本机版本）存在半精度 CrossEncoder 打分 bug（v6.0.0 修复，
    第三方实测 bf16 下 NDCG@10 0.1849 vs 0.6795，分数饱和）——必须用我们自己的
    模型/数据复现或证伪，不可直接落生产。

实验设计（零 API 成本，全本地）：
  A. reranker（bge-reranker-v2-m3, CrossEncoder）fp16 vs fp32：
     - 同一 (query, rrf_pool) 对集（来自今晨 details.json，fp32 基线同参数）
     - 指标：唯一分数数（饱和检测）、top5 排序一致率、rerank HIT@1/R@1、
       分数绝对漂移（max/mean）、RERANK_CONF_FLOOR=0.07 边际翻转数、延迟对比
  B. bge-m3（SentenceTransformer）fp16 vs fp32 全量重编码：
     - 指标：dense top20 重叠率、DENSE_OOD_FLOOR=0.55 边际翻转数、
       端到端管线 R@1（exact 短路 + BM25 + 加权 RRF k=20/w=1.5）、编码延迟

决策输出：三选一——直接落 / 绑 ST 6.x 升级 / 只落 bge-m3 侧。

用法：python eval/kb_v2/fp16_ab_test.py
"""
import json
import os
import sys
import time

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DETAILS = os.path.join(HERE, "eval_retrieval_rerank_details.json")
KB_PATH = os.path.join(PACKAGE_ROOT, "data", "knowledge_base_v2.json")  # 单源化：canonical=data/

RERANK_CONF_FLOOR = 0.07
DENSE_OOD_FLOOR = 0.55
RRF_K, RRF_W_DENSE = 20, 1.5
RECALL_WIDTH = 20

BGE_M3_PATH = os.getenv("BGE_M3_PATH", "")
BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", "")


def eval_ranks(titles, expected):
    hit1 = 1.0 if titles and titles[0] in expected else 0.0
    mrr = 0.0
    for i, t in enumerate(titles):
        if t in expected:
            mrr = 1.0 / (i + 1)
            break
    return hit1, mrr


def part_a_reranker(records, title2entry_flat):
    """A：reranker fp16 vs fp32 漂移实测。"""
    import torch
    from sentence_transformers import CrossEncoder
    from dense_retriever import DenseRetriever

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_pairs(rec):
        pairs, titles = [], []
        for t in rec["rrf_pool"]:
            e = title2entry_flat.get(t)
            if e is not None:
                pairs.append((rec["query"], DenseRetriever._entry_text(e)))
                titles.append(t)
        return pairs, titles

    all_pairs, all_titles = [], []
    for rec in records:
        p, t = build_pairs(rec)
        all_pairs.extend(p)
        all_titles.append(t)
    print(f"A. reranker 漂移实测：{len(records)} query × 池对共 {len(all_pairs)} 对\n")

    results = {}
    for dtype, label in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        model = CrossEncoder(
            BGE_RERANKER_PATH, device=device,
            model_kwargs={"torch_dtype": dtype},
        )
        # 预热一帧（排除首帧编译噪声）
        model.predict([("预热", "预热")], show_progress_bar=False)
        t0 = time.perf_counter()
        scores_all = model.predict(all_pairs, batch_size=32, show_progress_bar=False)
        dt = time.perf_counter() - t0
        results[label] = (scores_all, dt)
        del model
        torch.cuda.empty_cache()
        print(f"  [{label}] 推理 {len(all_pairs)} 对耗时 {dt:.1f}s")

    s32, t32 = results["fp32"]
    s16, t16 = results["fp16"]
    s32 = [float(x) for x in s32]
    s16 = [float(x) for x in s16]

    # 分数级漂移
    drifts = [abs(a - b) for a, b in zip(s32, s16)]
    uniq32, uniq16 = len(set(round(x, 4) for x in s32)), len(set(round(x, 4) for x in s16))
    print(f"\n  唯一分数数（4位小数）：fp32={uniq32} | fp16={uniq16}（饱和则 fp16 远小于 fp32）")
    print(f"  分数漂移：mean={sum(drifts)/len(drifts):.4f} max={max(drifts):.4f}")

    # 逐 query：排序一致率 / HIT@1 / 边际翻转
    idx, order_same, h32, h16, floor_flips, r1_32, r1_16 = 0, 0, 0.0, 0.0, 0, 0.0, 0.0
    for rec, titles in zip(records, all_titles):
        n = len(titles)
        sc32 = s32[idx:idx + n]
        sc16 = s16[idx:idx + n]
        idx += n
        exp = set(rec["expected"])
        top32 = [t for _, t in sorted(zip(sc32, titles), key=lambda x: x[0], reverse=True)][:5]
        top16 = [t for _, t in sorted(zip(sc16, titles), key=lambda x: x[0], reverse=True)][:5]
        if top32 == top16:
            order_same += 1
        a1, _ = eval_ranks(top32, exp)
        b1, _ = eval_ranks(top16, exp)
        h32 += a1
        h16 += b1
        # recall@1 口径（多期望除以个数）
        r1_32 += (1.0 / len(exp)) if top32 and top32[0] in exp else 0.0
        r1_16 += (1.0 / len(exp)) if top16 and top16[0] in exp else 0.0
        # 门禁边际翻转：top1 分数在 0.07 两侧变化
        f32 = max(sc32)
        f16 = max(sc16)
        if (f32 >= RERANK_CONF_FLOOR) != (f16 >= RERANK_CONF_FLOOR):
            floor_flips += 1
    n = len(records)
    print(f"  top5 排序完全一致率：{order_same}/{n} = {order_same/n:.1%}")
    print(f"  rerank HIT@1：fp32={h32/n:.3f} | fp16={h16/n:.3f}（Δ{(h16-h32)/n:+.3f}）")
    print(f"  rerank R@1：fp32={r1_32/n:.3f} | fp16={r1_16/n:.3f}（Δ{(r1_16-r1_32)/n:+.3f}）")
    print(f"  RERANK_CONF_FLOOR={RERANK_CONF_FLOOR} 边际翻转：{floor_flips} 条")
    print(f"  速度：fp16/fp32 = {t16/t32:.2f}x")
    return h16 / n, order_same / n, floor_flips


def part_b_dense(records):
    """B：bge-m3 fp16 vs fp32 全量重编码漂移。"""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from kb_retriever import KBRetriever, _expand_query, _tokenize
    from rank_bm25 import BM25Okapi
    from dense_retriever import DenseRetriever

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    domains = {d: es for d, es in kb.items() if not d.startswith("_")}
    entries_flat = {e["title"]: e for es in domains.values() for e in es}
    texts = [DenseRetriever._entry_text(e) for es in domains.values() for e in es]
    title_order = [e["title"] for es in domains.values() for e in es]
    queries = [r["query"] for r in records]

    print(f"\nB. bge-m3 漂移实测：KB {len(texts)} 条 + query {len(queries)} 条全量重编码\n")

    def encode_all(dtype):
        model = SentenceTransformer(BGE_M3_PATH, device=device,
                                    model_kwargs={"torch_dtype": dtype})
        model.encode(["预热"], normalize_embeddings=True)
        t0 = time.perf_counter()
        kb_vec = model.encode(texts, batch_size=32, normalize_embeddings=True,
                              show_progress_bar=False)
        t_kb = time.perf_counter() - t0
        t0 = time.perf_counter()
        q_vec = model.encode(queries, batch_size=32, normalize_embeddings=True,
                             show_progress_bar=False)
        t_q = time.perf_counter() - t0
        del model
        torch.cuda.empty_cache()
        return np.asarray(kb_vec), np.asarray(q_vec), t_kb, t_q

    kb32, q32, tkb32, tq32 = encode_all(torch.float32)
    kb16, q16, tkb16, tq16 = encode_all(torch.float16)
    print(f"  编码延迟：KB fp32 {tkb32:.1f}s vs fp16 {tkb16:.1f}s（{tkb16/tkb32:.2f}x）| "
          f"query fp32 {tq32:.1f}s vs fp16 {tq16:.1f}s（{tq16/tq32:.2f}x）")

    # title -> 向量索引（按 domain 分组检索）
    dom_titles = {d: [e["title"] for e in es] for d, es in domains.items()}
    title_idx = {t: i for i, t in enumerate(title_order)}

    def dense_hits(dtype_kb, dtype_q, domain, query, top_k=RECALL_WIDTH):
        idxs = [title_idx[t] for t in dom_titles[domain]]
        sims = dtype_kb[idxs] @ dtype_q[queries.index(query)]
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), dom_titles[domain][i]) for i in order]

    # BM25 原始分（与 ablation_fusion 同口径）
    bm25_idx = {}
    for d, es in domains.items():
        def etext(e):
            parts = [e.get("category", ""), e.get("title", ""), e.get("content", "")]
            kws = e.get("keywords", [])
            if isinstance(kws, list):
                parts.extend(str(k) for k in kws)
            return " ".join(p for p in parts if p)
        bm25_idx[d] = BM25Okapi([_tokenize(etext(e)) for e in es])

    def bm25_hits(domain, expanded, top_k=RECALL_WIDTH):
        es = domains[domain]
        scores = bm25_idx[domain].get_scores(_tokenize(expanded))
        ranked = sorted(((scores[i], es[i].get("title", "")) for i in range(len(es))),
                        key=lambda x: x[0], reverse=True)[:top_k]
        return [(s, t) for s, t in ranked if s > 0]

    def weighted_rrf(bm25_h, dense_h, top_k=5):
        sc = {}
        for r, (_, t) in enumerate(bm25_h):
            sc[t] = sc.get(t, 0.0) + 1.0 / (RRF_K + r + 1)
        for r, (_, t) in enumerate(dense_h):
            sc[t] = sc.get(t, 0.0) + RRF_W_DENSE / (RRF_K + r + 1)
        return [t for t, _ in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    prod = KBRetriever(kb_path=KB_PATH)
    overlap_sum, floor_flips, hit32, hit16, n_fused = 0, 0, 0.0, 0.0, 0
    for rec in records:
        q, domain, exp = rec["query"], rec["domain"], set(rec["expected"])
        expanded = _expand_query(q)
        exact = prod._exact_hits(domain, expanded)
        if exact:
            titles = [str(e.get("title") or "") for e in exact]
            a1, _ = eval_ranks(titles, exp)
            b1, _ = eval_ranks(titles, exp)
            hit32 += a1
            hit16 += b1
            continue
        n_fused += 1
        b = bm25_hits(domain, expanded)
        d32 = dense_hits(kb32, q32, domain, q)
        d16 = dense_hits(kb16, q16, domain, q)
        set32, set16 = {t for _, t in d32}, {t for _, t in d16}
        overlap_sum += len(set32 & set16) / RECALL_WIDTH
        # DENSE_OOD_FLOOR 边际翻转（dense top1 分数两侧变化）
        f32 = d32[0][0] if d32 else 0.0
        f16 = d16[0][0] if d16 else 0.0
        if (f32 >= DENSE_OOD_FLOOR) != (f16 >= DENSE_OOD_FLOOR):
            floor_flips += 1
        t32 = weighted_rrf(b, d32)
        t16 = weighted_rrf(b, d16)
        a1, _ = eval_ranks(t32, exp)
        b1, _ = eval_ranks(t16, exp)
        hit32 += a1
        hit16 += b1

    n = len(records)
    print(f"  dense top20 重叠率（融合路 {n_fused} 条均值）：{overlap_sum/n_fused:.1%}")
    print(f"  DENSE_OOD_FLOOR={DENSE_OOD_FLOOR} 边际翻转：{floor_flips} 条")
    print(f"  管线 HIT@1（exact短路+BM25+加权RRF）：fp32={hit32/n:.3f} | fp16={hit16/n:.3f}（Δ{(hit16-hit32)/n:+.3f}）")


def main() -> None:
    records = json.load(open(DETAILS, encoding="utf-8"))
    kb = json.load(open(KB_PATH, encoding="utf-8"))
    title2entry_flat = {e["title"]: e for d, es in kb.items() if not d.startswith("_") for e in es}
    print(f"基座：{len(records)} 条金标（今晨 details，k=20/w=1.5）\n")
    part_a_reranker(records, title2entry_flat)
    part_b_dense(records)
    print("\n判定参考：fp16 rerank HIT@1 漂移 <0.5pp 且 floor 翻转 ≤2 条 → 可落；"
          "否则 reranker fp16 绑 ST 6.x 升级；bge-m3 侧单独按 B 部分数据判定")


if __name__ == "__main__":
    main()
