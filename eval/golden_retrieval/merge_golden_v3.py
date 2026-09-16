"""A2-4 合并新增金标到主 CSV（幂等：重复运行不产生重复 id/query）。

主 CSV 原无 id 列（仅 7 列），此处统一补齐 id（gr-001 起，TEMPLATE 规范）。
校验：id 唯一、query 唯一、title 存在性复核、域/难度枚举。
用法：python eval/golden_retrieval/merge_golden_v3.py
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "golden_retrieval_golden_v2.csv")
ADD = os.path.join(HERE, "golden_retrieval_golden_v2_add.csv")
TITLES = os.path.join(HERE, "kb_v2_titles.json")

kb_titles = json.load(open(TITLES, encoding="utf-8"))
valid = set()
for v in kb_titles.values():
    valid.update(v)

main_rows = list(csv.DictReader(open(MAIN, encoding="utf-8-sig")))
add_rows = list(csv.DictReader(open(ADD, encoding="utf-8-sig")))

# 主行补 id（按当前顺序 gr-001 起；若已有 id 保留）
if "id" not in main_rows[0]:
    for i, r in enumerate(main_rows, start=1):
        r["id"] = f"gr-{i:03d}"
    main_rows[0]["note"] = ""

# 列对齐：主行缺列补空
all_keys = ["id", "query", "domain", "expected_titles", "difficulty", "source", "suggest", "candidates", "note"]
for r in main_rows:
    for k in all_keys:
        r.setdefault(k, "")

main_ids = {r["id"] for r in main_rows}
main_queries = {r["query"] for r in main_rows}

added = 0
for r in add_rows:
    if r["id"] in main_ids:
        continue  # 已合并过（幂等）
    if r["query"] in main_queries:
        print(f"[WARN] query 已存在，跳过: {r['query']}")
        continue
    for k in all_keys:
        r.setdefault(k, "")
    main_rows.append(r)
    main_ids.add(r["id"])
    main_queries.add(r["query"])
    added += 1

# 校验
all_ids = [r["id"] for r in main_rows]
all_queries = [r["query"] for r in main_rows]
assert len(set(all_ids)) == len(all_ids), "id 重复！"
assert len(set(all_queries)) == len(all_queries), "query 重复！"
assert all(r["difficulty"] in ("easy", "hard", "ambiguous") for r in main_rows), "difficulty 非法"
assert all(r["domain"] in ("product", "tech", "billing", "general", "complaint") for r in main_rows), "domain 非法"
for r in main_rows:
    for t in r["expected_titles"].split(";"):
        t = t.strip()
        if t and t not in valid:
            print(f"[MISSING] {r['id']} | {r['domain']} | {t}")
            sys.exit(1)

with open(MAIN, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_keys)
    w.writeheader()
    w.writerows(main_rows)

print(f"合并完成：新增 {added} 条，主 CSV 现有 {len(main_rows)} 条")
from collections import Counter
print("domain:", dict(Counter(r["domain"] for r in main_rows)))
print("difficulty:", dict(Counter(r["difficulty"] for r in main_rows)))
print("source:", dict(Counter(r["source"] for r in main_rows)))
