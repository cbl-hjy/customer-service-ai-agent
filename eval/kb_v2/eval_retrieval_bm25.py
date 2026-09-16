#!/usr/bin/env python3
"""检索层基线评估器：BM25-only 在 KB v2 + 金标集 v2 上的 Recall@K/MRR/nDCG。

- 按 difficulty 分组统计（easy/hard/ambiguous 分开报）——hard/ambiguous 是
  混合检索/重排消融的区分度来源，分组看才能量化各档提升
- 与 kb_retriever 共用同一检索逻辑（retrieve 返回格式化文本，retrieve_top_title
  返回 top-1 标题）——评估看到的就是 agent 检索到的
- 支持多标（expected_titles 分号分隔）：命中任一即算 hit；Recall@K 按命中数/K

用法：python eval/kb_v2/eval_retrieval_bm25.py
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

# 需要 top-K 列表 → 直接调用内部方法（与 retrieve() 同判定）
from kb_retriever import _expand_query  # noqa: E402

K_VALUES = (1, 3, 5)


def dcg(relevances: list) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def idcg(num_rel: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i in range(num_rel))


def main() -> None:
    rows = list(csv.DictReader(open(GOLDEN_V2, encoding="utf-8-sig")))
    # 显式加载 v2 KB（KBRetriever 接受 kb_path）
    from kb_retriever import KBRetriever  # noqa: E402
    retriever = KBRetriever(kb_path=V2_KB)

    # 统计容器
    stats = {d: {"n": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr": 0.0, "ndcg@5": 0.0, "hit@1": 0}
             for d in ["easy", "hard", "ambiguous", "ALL"]}

    details = []
    for r in rows:
        query, domain = r["query"], r["domain"]
        expected = [t for t in r["expected_titles"].split(";") if t]
        diff = r["difficulty"]
        if domain not in retriever._data:
            print(f"[skip] 域不存在: {domain} ({query})")
            continue

        # 检索 top-5 标题（与 retrieve 相同判定路径：查询扩展 → 精确预检 → BM25）
        expanded = _expand_query(query)
        exact = retriever._exact_hits(domain, expanded)
        titles_ranked = [e.get("title", "") for e in exact]
        if not titles_ranked:
            bm25 = retriever._bm25_hits(domain, expanded, top_k=5)
            titles_ranked = [e.get("title", "") for e in bm25]

        # 指标
        hits = [1 if t in expected else 0 for t in titles_ranked]
        recall_at = {}
        for k in K_VALUES:
            topk = hits[:k]
            recall_at[k] = sum(topk) / len(expected) if expected else 0.0
        # MRR
        mrr = 0.0
        for i, t in enumerate(titles_ranked):
            if t in expected:
                mrr = 1.0 / (i + 1)
                break
        # nDCG@5
        rel = [1 if t in expected else 0 for t in titles_ranked[:5]]
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

        details.append({
            "query": query, "domain": domain, "diff": diff,
            "expected": expected, "retrieved_top5": titles_ranked,
            "recall@1": recall_at[1], "recall@3": recall_at[3], "mrr": round(mrr, 3),
        })

    print(f"=== BM25-only 基线（KB v2 {sum(len(v) for v in retriever._data.values())} 条 / 金标集 {len(rows)} 条）===")
    print(f"{'分组':<10}{'n':>4}{'R@1':>8}{'R@3':>8}{'R@5':>8}{'MRR':>8}{'nDCG@5':>8}{'HIT@1':>8}")
    for bucket in ["easy", "hard", "ambiguous", "ALL"]:
        s = stats[bucket]
        n = max(s["n"], 1)
        print(f"{bucket:<10}{s['n']:>4}{s['recall@1']/n:>8.3f}{s['recall@3']/n:>8.3f}{s['recall@5']/n:>8.3f}"
              f"{s['mrr']/n:>8.3f}{s['ndcg@5']/n:>8.3f}{s['hit@1']/n:>8.3f}")

    # 保存明细供消融对比
    out = os.path.join(HERE, "eval_retrieval_bm25_details.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=1)
    print(f"\n明细已存: {out}（供后续混合检索/重排消融逐条对比）")


if __name__ == "__main__":
    main()
