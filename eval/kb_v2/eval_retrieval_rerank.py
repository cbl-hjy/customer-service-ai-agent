#!/usr/bin/env python3
"""P2 重排评估：BM25-only 基线 vs 混合 RRF vs 混合 RRF + bge-reranker-v2-m3。

三路同金标集（95 条）、同指标口径（R@1/3/5、MRR、nDCG@5、HIT@1），按 difficulty 分组 + ALL。
reranker 路：RRF 融合宽池 top-20 → (query, 条目文本) CrossEncoder 打分 → 精排取 top-5。

用法：python eval/kb_v2/eval_retrieval_rerank.py
"""
import csv
import json
import math
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_V2 = os.path.join(os.path.dirname(HERE), "golden_retrieval", "golden_retrieval_golden_v2.csv")
V2_KB = os.path.join(HERE, "knowledge_base_v2.json")

from kb_retriever import KBRetriever, _expand_query  # noqa: E402
from dense_retriever import build_dense_index, DenseRetriever  # noqa: E402

# bge-reranker-v2-m3 本地快照路径（环境变量注入，不入库；2026-08-16 确认 2.2G 可加载，smoke PASS）
BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", "")

K_VALUES = (1, 3, 5)
# 2026-09-16 实验B消融定版：k=20 + 稠密路加权 1.5（与 hybrid_retriever 同口径，改一处必同步另一处）
RRF_K = 20
RRF_W_DENSE = 1.5
BM25_TOP_K = 20
RERANK_POOL = 20   # reranker 输入池：RRF 融合后取前 20
RERANK_TOP_K = 5   # reranker 输出


def rrf_fuse(bm25_hits: list, dense_hits: list, top_k: int = 5) -> list:
    """加权 RRF 融合：输入 [(score, title)]，输出前 top_k 个 title（与 hybrid_retriever 同口径）。"""
    scores: dict = {}
    for rank, (_, title) in enumerate(bm25_hits):
        scores[title] = scores.get(title, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (_, title) in enumerate(dense_hits):
        scores[title] = scores.get(title, 0.0) + RRF_W_DENSE / (RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_k]]


def dcg(relevances: list) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def idcg(num_rel: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i in range(num_rel))


def eval_ranks(titles: list, expected: list):
    """给定有序 title 列表，返回 recall_at / mrr / ndcg@5。"""
    hits = [1 if t in expected else 0 for t in titles]
    recall_at = {}
    for k in K_VALUES:
        recall_at[k] = sum(hits[:k]) / len(expected) if expected else 0.0
    mrr = 0.0
    for i, t in enumerate(titles):
        if t in expected:
            mrr = 1.0 / (i + 1)
            break
    rel = [1 if t in expected else 0 for t in titles[:5]]
    ndcg = dcg(rel) / idcg(len(expected)) if idcg(len(expected)) > 0 else 0.0
    return recall_at, mrr, ndcg, hits


def new_stats():
    return {d: {"n": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr": 0.0, "ndcg@5": 0.0, "hit@1": 0}
            for d in ["easy", "hard", "ambiguous", "ALL"]}


def accumulate(stats, diff, recall_at, mrr, ndcg, hits):
    for bucket in [diff, "ALL"]:
        s = stats[bucket]
        s["n"] += 1
        s["recall@1"] += recall_at[1]
        s["recall@3"] += recall_at[3]
        s["recall@5"] += recall_at[5]
        s["mrr"] += mrr
        s["ndcg@5"] += ndcg
        s["hit@1"] += 1 if (hits and hits[0]) else 0


def print_table(title: str, stats: dict):
    print(f"--- {title} ---")
    print(f"{'分组':<10}{'n':>4}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'nDCG@5':>8}{'HIT@1':>8}")
    for bucket in ["easy", "hard", "ambiguous", "ALL"]:
        s = stats[bucket]
        n = max(s["n"], 1)
        print(f"{bucket:<10}{s['n']:>4}{s['recall@1']/n:>8.3f}{s['recall@3']/n:>8.3f}{s['recall@5']/n:>8.3f}"
              f"{s['mrr']/n:>8.3f}{s['ndcg@5']/n:>8.3f}{s['hit@1']/n:>8.3f}")


def main() -> None:
    rows = list(csv.DictReader(open(GOLDEN_V2, encoding="utf-8-sig")))
    bm25 = KBRetriever(kb_path=V2_KB)
    dense = build_dense_index(V2_KB)

    from sentence_transformers import CrossEncoder
    import torch
    reranker = CrossEncoder(BGE_RERANKER_PATH,
                            device="cuda" if torch.cuda.is_available() else "cpu")

    # domain 内 title → entry 映射（reranker 打分需要完整条目文本）
    title2entry = {}
    for domain, entries in bm25._data.items():
        title2entry[domain] = {e.get("title", ""): e for e in entries}

    stats_bm25 = new_stats()
    stats_rrf = new_stats()
    stats_rr = new_stats()
    details = []

    for i, r in enumerate(rows):
        query, domain = r["query"], r["domain"]
        expected = [t for t in r["expected_titles"].split(";") if t]
        diff = r["difficulty"]
        if domain not in bm25._data or domain not in dense._vectors:
            print(f"[skip] 域缺失: {domain} ({query})")
            continue

        expanded = _expand_query(query)
        exact = bm25._exact_hits(domain, expanded)
        if exact:
            bm25_ranked = [(1.0, e.get("title", "")) for e in exact]
        else:
            bm25_hits = bm25._bm25_hits(domain, expanded, top_k=BM25_TOP_K)
            bm25_ranked = [(1.0, e.get("title", "")) for e in bm25_hits]
        dense_ranked = dense.search(domain, query, top_k=20)

        # 路 1：BM25-only top-5（基线，与 eval_retrieval_bm25.py 同判定）
        bm25_top5 = [t for _, t in bm25_ranked][:5]
        # 路 2：混合 RRF top-5（P1 现状）
        rrf_top5 = rrf_fuse(bm25_ranked, dense_ranked, top_k=5)
        # 路 3：RRF 宽池 top-20 → reranker 精排 top-5
        rrf_pool = rrf_fuse(bm25_ranked, dense_ranked, top_k=RERANK_POOL)
        t2e = title2entry[domain]
        pool_pairs = [(query, DenseRetriever._entry_text(t2e[t])) for t in rrf_pool if t in t2e]
        pool_titles = [t for t in rrf_pool if t in t2e]
        rr_scores = reranker.predict(pool_pairs, batch_size=32, show_progress_bar=False)
        rr_top5 = [t for _, t in sorted(zip(rr_scores, pool_titles), key=lambda x: x[0], reverse=True)][:RERANK_TOP_K]

        ra1, mrr1, ndcg1, hits1 = eval_ranks(bm25_top5, expected)
        ra2, mrr2, ndcg2, hits2 = eval_ranks(rrf_top5, expected)
        ra3, mrr3, ndcg3, hits3 = eval_ranks(rr_top5, expected)
        accumulate(stats_bm25, diff, ra1, mrr1, ndcg1, hits1)
        accumulate(stats_rrf, diff, ra2, mrr2, ndcg2, hits2)
        accumulate(stats_rr, diff, ra3, mrr3, ndcg3, hits3)

        details.append({
            "query": query, "domain": domain, "diff": diff, "expected": expected,
            "bm25_top5": bm25_top5, "rrf_top5": rrf_top5, "rerank_top5": rr_top5,
            "rrf_pool": rrf_pool, "rr_scores": [round(float(s), 4) for s in rr_scores],
            "recall@1": {"bm25": ra1[1], "rrf": ra2[1], "rerank": ra3[1]},
            "mrr": {"bm25": round(mrr1, 3), "rrf": round(mrr2, 3), "rerank": round(mrr3, 3)},
        })

        if (i + 1) % 20 == 0:
            print(f"[进度] {i + 1}/{len(rows)}", flush=True)

    print(f"\n=== P2 重排评估 金标集 {len(rows)} 条 ===")
    print_table("BM25-only 基线", stats_bm25)
    print()
    print_table("混合 RRF（P1 现状）", stats_rrf)
    print()
    print_table("混合 RRF + bge-reranker-v2-m3", stats_rr)

    # 对比汇总（ALL）
    print("\n=== ALL 对比 ===")
    print(f"{'路':<22}{'R@1':>8}{'R@3':>8}{'MRR':>8}{'nDCG@5':>8}{'HIT@1':>8}")
    for name, st in [("BM25-only", stats_bm25), ("RRF", stats_rrf), ("RRF+rerank", stats_rr)]:
        s = st["ALL"]
        n = max(s["n"], 1)
        print(f"{name:<22}{s['recall@1']/n:>8.3f}{s['recall@3']/n:>8.3f}{s['mrr']/n:>8.3f}"
              f"{s['ndcg@5']/n:>8.3f}{s['hit@1']/n:>8.3f}")

    out = os.path.join(HERE, "eval_retrieval_rerank_details.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"\n明细已写: {out}")


if __name__ == "__main__":
    main()
