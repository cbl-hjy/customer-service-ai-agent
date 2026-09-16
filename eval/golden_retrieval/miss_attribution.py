#!/usr/bin/env python3
"""ECD miss 归因分析：43 条未命中 query 逐条跨 5 域检查检索表现。

三类根因：
  ① COVERAGE_GAP  KB 无相关条目（该域 top-1 分数 < 0.05，语义无对应）→ 扩库解决
  ② THRESHOLD     KB 有候选但 BM25 分数被 SCORE_THRESHOLD(0.30) 过滤 → 调阈值/混合检索
  ③ ALGO_MISS     KB 有语义对应条目但 BM25 完全没召回（top 分数 ≈0）→ 混合检索解决

输出：归因统计 + 明细（含跨域 top-3 分数），供人工复核。
用法：python eval/golden_retrieval/miss_attribution.py
"""
import csv
import os
import sys

# Windows 控制台/重定向 GBK 兜底：强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

from kb_retriever import get_retriever, SCORE_THRESHOLD  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(HERE, "ecd_seeds.csv")

DOMAINS = ["product", "tech", "billing", "general", "complaint"]
# 种子 domain_hint → 检索域（粗分可能错，归因时跨全域查）
MISSING = []  # (hint, query) 由上一轮基线跑出


def collect_missing() -> list:
    retriever = get_retriever()
    rows = list(csv.DictReader(open(SEEDS, encoding="utf-8")))
    out = []
    for r in rows:
        q = r["query"]
        hint = r["domain_hint"]
        if not retriever.retrieve(hint if hint in DOMAINS else "general", q):
            out.append((hint, q))
    return out


def classify_miss(top_score: float) -> str:
    """按跨域 top-1 分数归因：
    <0.05  KB 无相关条目 → COVERAGE_GAP（扩库解决）
    0.05~0.30  有候选但被阈值过滤 → THRESHOLD（调阈值/混合检索）
    >=0.30  有高分候选 → ALGO_MISS（查错域或词不匹配，需人工复核）
    """
    if top_score < 0.05:
        return "COVERAGE_GAP"
    if top_score < SCORE_THRESHOLD:
        return "THRESHOLD"
    return "ALGO_MISS"


def main() -> None:
    retriever = get_retriever()
    missing = collect_missing()
    stats = {"COVERAGE_GAP": 0, "THRESHOLD": 0, "ALGO_MISS": 0}
    detail = []
    for hint, q in missing:
        # 跨 5 域查 top-3 分数
        best = []  # (score, domain, title)
        for dom in DOMAINS:
            index = retriever._indexes.get(dom)
            if index is None:
                continue
            q_toks = retriever._tokenize(q)
            if not q_toks:
                continue
            scores = index.get_scores(q_toks)
            entries = retriever._data.get(dom, [])
            ranked = sorted(
                ((scores[i], entries[i].get("title", "")) for i in range(len(entries))),
                key=lambda x: x[0], reverse=True,
            )[:3]
            for s, t in ranked:
                best.append((round(s, 3), dom, t))
        best.sort(key=lambda x: x[0], reverse=True)
        top_score = best[0][0] if best else 0.0

        # 归因判定
        reason = classify_miss(top_score)
        stats[reason] += 1
        detail.append((reason, hint, q, best[:3]))

    print(f"=== ECD miss 归因：{len(missing)} 条 ===")
    for k, v in stats.items():
        pct = v / len(missing) * 100 if missing else 0
        print(f"  {k}: {v}（{pct:.1f}%）")
    print()
    for reason, hint, q, best in detail:
        top_str = " | ".join(f"{d}:{t}({s})" for s, d, t in best) if best else "无任何命中"
        print(f"[{reason}] ({hint}) {q}")
        print(f"    top: {top_str}")


if __name__ == "__main__":
    main()
