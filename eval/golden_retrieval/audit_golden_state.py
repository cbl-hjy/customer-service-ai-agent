"""A2-1 盘点工具（一次性）：金标 95 条分布 + KB 601 各域 title 清单导出。

产出：
  1. 控制台：金标 domain/difficulty/source 分布
  2. eval/golden_retrieval/kb_v2_titles.json：各域 title 全集（供标注对齐 expected_titles）
用法：python eval/golden_retrieval/audit_golden_state.py
"""
import csv
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_retrieval_golden_v2.csv")
KB = os.path.join(os.path.dirname(HERE), "kb_v2", "knowledge_base_v2.json")
OUT = os.path.join(HERE, "kb_v2_titles.json")

rows = list(csv.DictReader(open(GOLDEN, encoding="utf-8-sig")))
print(f"金标总数: {len(rows)}")
print("domain 分布:", dict(Counter(r["domain"] for r in rows)))
print("difficulty 分布:", dict(Counter(r["difficulty"] for r in rows)))
print("source 分布:", dict(Counter(r["source"] for r in rows)))

# 难度×域交叉
cross = Counter((r["domain"], r["difficulty"]) for r in rows)
for k in sorted(cross):
    print(f"  {k[0]:10s} {k[1]:10s} {cross[k]}")

# KB 各域 title 清单
kb = json.load(open(KB, encoding="utf-8"))
titles = {}
for dom, entries in kb.items():
    titles[dom] = [e.get("title", "") for e in entries]
    print(f"KB[{dom}] 条目 {len(entries)}")

json.dump(titles, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"title 清单已导出: {OUT}")

# 现有金标 expected_titles 与 KB title 的对齐检查（重名/缺失）
for dom in titles:
    valid = set(titles[dom])
    for r in rows:
        if r["domain"] != dom:
            continue
        for t in r["expected_titles"].split(";"):
            t = t.strip()
            if t and t not in valid:
                print(f"  [缺失或已改] {r['id']} 期望title不在{dom}域: {t}")
