#!/usr/bin/env python3
"""P1 混合检索评估：BM25 + bge-m3 稠密 → RRF 融合，与 BM25-only 基线对比。

指标口径与 eval_retrieval_bm25.py 完全一致（Recall@1/3/5、MRR、nDCG@5、HIT@1），
按 difficulty 分组 + ALL。输出对比表 + 逐条明细（JSON），供复盘分析。

RRF 融合：score(q,d) = Σ_r 1/(k + rank_r(d))，k=60（惯例值）
两路各取 top-20（稠密 DENSE_TOP_K=20，BM25 放宽到 20），融合后取 top-5。

用法：python eval/kb_v2/eval_retrieval_hybrid.py
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
V2_KB = os.path.join(PACKAGE_ROOT, "data", "knowledge_base_v2.json")  # 单源化：canonical=data/

from kb_retriever import KBRetriever, _expand_query, SCORE_THRESHOLD  # noqa: E402
from dense_retriever import build_dense_index  # noqa: E402

K_VALUES = (1, 3, 5)
RRF_K = 60
BM25_TOP_K = 20  # 融合前放宽 BM25 召回（原 TOP_K=3 太窄，RRF 需要宽池）


def rrf_fuse(bm25_hits: list, dense_hits: list, top_k: int = 5) -> list:
    """RRF 融合：输入 [(score, title)]，输出前 top_k 个 title。"""
    scores: dict = {}
    for rank, (_, title) in enumerate(bm25_hits):
        scores[title] = scores.get(title, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (_, title) in enumerate(dense_hits):
        scores[title] = scores.get(title, 0.0) + 1.0 / (RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_k]]


def dcg(relevances: list) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def idcg(num_rel: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i in range(num_rel))


def main() -> None:
    rows = list(csv.DictReader(open(GOLDEN_V2, encoding="utf-8-sig")))
    bm25 = KBRetriever(kb_path=V2_KB)
    dense = build_dense_index(V2_KB)

    stats = {d: {"n": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr": 0.0, "ndcg@5": 0.0, "hit@1": 0}
             for d in ["easy", "hard", "ambiguous", "ALL"]}
    details = []

    for r in rows:
        query, domain = r["query"], r["domain"]
        expected = [t for t in r["expected_titles"].split(";") if t]
        diff = r["difficulty"]
        if domain not in bm25._data or domain not in dense._vectors:
            print(f"[skip] 域缺失: {domain} ({query})")
            continue

        expanded = _expand_query(query)
        # BM25 路：精确预检优先（与基线同判定），否则放宽 top-20
        exact = bm25._exact_hits(domain, expanded)
        if exact:
            bm25_ranked = [(1.0, e.get("title", "")) for e in exact]
        else:
            bm25_hits = bm25._bm25_hits(domain, expanded, top_k=BM25_TOP_K)
            bm25_ranked = [(1.0, e.get("title", "")) for e in bm25_hits]
        # 稠密路：top-20
        dense_ranked = dense.search(domain, query, top_k=20)

        fused_titles = rrf_fuse(bm25_ranked, dense_ranked, top_k=5)

        hits = [1 if t in expected else 0 for t in fused_titles]
        recall_at = {}
        for k in K_VALUES:
            recall_at[k] = sum(hits[:k]) / len(expected) if expected else 0.0
        mrr = 0.0
        for i, t in enumerate(fused_titles):
            if t in expected:
                mrr = 1.0 / (i + 1)
                break
        rel = [1 if t in expected else 0 for t in fused_titles[:5]]
        ndcg = dcg(rel) / idcg(len(expected)) if idcg(len(expected)) > 0 else 0.0

        for bucket in [diff, "ALL"]:
            s = stats[bucket]
            s["n"] += 1
            s["recall@1"] += recall_at[1]
            s["recall@3"] += recall_at[3]
            s["recall@5"] += recall_at[5]
            s["mrr"] += mrr
            s["ndcg@5"] += ndcg
            s["hit@1"] += 1 if (hits and hits[0]) else 0

        details.append({"query": query, "domain": domain, "diff": diff,
                        "expected": expected, "retrieved_top5": fused_titles,
                        "recall@1": recall_at[1], "mrr": round(mrr, 3)})

    print(f"=== P1 混合检索（BM25 + bge-m3 RRF） 金标集 {len(rows)} 条 ===")
    print(f"{'分组':<10}{'n':>4}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'nDCG@5':>8}{'HIT@1':>8}")
    for bucket in ["easy", "hard", "ambiguous", "ALL"]:
        s = stats[bucket]
        n = max(s["n"], 1)
        print(f"{bucket:<10}{s['n']:>4}{s['recall@1']/n:>8.3f}{s['recall@3']/n:>8.3f}{s['recall@5']/n:>8.3f}"
              f"{s['mrr']/n:>8.3f}{s['ndcg@5']/n:>8.3f}{s['hit@1']/n:>8.3f}")

    out = os.path.join(HERE, "eval_retrieval_hybrid_details.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1)
    print(f"\n明细已存: {out}")


if __name__ == "__main__":
    main()
