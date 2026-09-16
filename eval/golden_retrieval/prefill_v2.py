#!/usr/bin/env python3
"""KB v2 就位后重跑 golden 预填：给原 DROP? 的 63 条用 v2 找对应条目。

对每条 query：
- 用 KB v2 跨 5 域检索，输出 top-3 候选
- 复用 classify_suggest 判定 KEEP/DROP?（v2 覆盖面大，很多原 DROP? 会转 KEEP）

用法：python eval/golden_retrieval/prefill_v2.py
输出：eval/golden_retrieval/golden_retrieval_prefill_v2.csv（含原金标集 83 条 + 新可标条目）
"""
import csv
import json
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
V2_KB = os.path.join(PACKAGE_ROOT, "data", "knowledge_base_v2.json")  # 单源化：canonical=data/
GOLDEN = os.path.join(HERE, "golden_retrieval_golden.csv")
PREFILL_V1 = os.path.join(HERE, "golden_retrieval_prefill.csv")
OUT = os.path.join(HERE, "golden_retrieval_prefill_v2.csv")

DOMAINS = ["product", "tech", "billing", "general", "complaint"]


def search_v2(retriever, kb, q: str):
    """跨 5 域检索 KB v2（v2 结构同 v1：{domain: [entries]}）。"""
    best = []
    for dom in DOMAINS:
        entries = kb.get(dom, [])
        if not entries:
            continue
        # 简单词重叠打分（BM25 语义近似；v2 条目多，用关键词匹配足够给候选）
        q_toks = set(retriever._tokenize(q))
        scored = []
        for e in entries:
            text = e.get("title", "") + " " + " ".join(e.get("keywords", []))
            e_toks = set(retriever._tokenize(text))
            inter = len(q_toks & e_toks)
            if inter > 0:
                scored.append((inter, e.get("title", "")))
        scored.sort(key=lambda x: x[0], reverse=True)
        for s, t in scored[:3]:
            best.append((s, dom, t))
    best.sort(key=lambda x: x[0], reverse=True)
    return best[:6]


def main() -> None:
    from kb_retriever import get_retriever  # noqa: E402
    retriever = get_retriever()
    kb = json.load(open(V2_KB, encoding="utf-8"))

    # 收集所有 query：原 DROP? 63 条（来自 prefill_v1）+ 原金标集 83 条（保持不动）
    prefill_v1 = list(csv.DictReader(open(PREFILL_V1, encoding="utf-8-sig")))
    golden = list(csv.DictReader(open(GOLDEN, encoding="utf-8-sig")))

    # 金标集已有（不再处理）
    golden_queries = {r["query"] for r in golden}
    # 待补标 query = prefill_v1 全部 - 金标集已有（含我改判 DROP? 的 30 条，
    # 它们在 golden.csv 被排除但 prefill_v1.csv 的 suggest 字段未同步更新）
    drop_queries = {r["query"] for r in prefill_v1} - golden_queries

    rows = []
    # 1) 金标集原样保留
    for r in golden:
        rows.append(r)
    # 2) DROP? 用 v2 重查
    new_keeps = 0
    for r in prefill_v1:
        if r["query"] not in drop_queries:
            continue
        best = search_v2(retriever, kb, r["query"])
        has_cand = any(s >= 1 for s, _, _ in best)
        cand_str = " | ".join(f"{d}:{t}({s})" for s, d, t in best) if best else "无候选"
        if has_cand:
            new_keeps += 1
            rows.append({
                "query": r["query"], "domain": r["domain"], "expected_titles": "",
                "difficulty": "hard", "source": r["source"], "suggest": "KEEP",
                "candidates": cand_str,
            })

    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"预填 v2 完成：{len(rows)} 条 → {OUT}")
    print(f"  原金标集保留 {len(golden)} 条")
    print(f"  原 DROP? {len(drop_queries)} 条中用 v2 找到候选 {new_keeps} 条（待人工补标 expected_titles）")
    print(f"  仍无对应 {len(drop_queries) - new_keeps} 条（保留 DROP?）")


if __name__ == "__main__":
    main()
