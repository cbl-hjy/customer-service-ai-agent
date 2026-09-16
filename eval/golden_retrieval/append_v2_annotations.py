#!/usr/bin/env python3
"""将 v2 补标的 12 条追加进正式金标集 golden_retrieval_golden.csv。

判断标准（客服专业认知 + v2 条目存在性）：
- query 问的事，v2 里有能直接回答的条目 → KEEP + expected_titles（域内真实标题）
- 母婴/食品问法（核桃/湿巾/茶具）或无语义问法 → 不补（KB 无对应，保留在 prefill 的 DROP 集合）

用法：python eval/golden_retrieval/append_v2_annotations.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_retrieval_golden.csv")

# (query, domain, expected_titles, difficulty)
NEW_ANNOTATIONS = [
    ("为什么有运费", "general", "运费计算规则", "easy"),
    ("买多少斤包邮", "general", "包邮条件", "easy"),
    ("价格可以有优惠吗", "billing", "会员价规则", "hard"),
    ("发中通圆通吗", "general", "快递选择", "easy"),
    ("取消重新加双袜子重新拍你说不用", "general", "订单取消", "hard"),
    ("我买5个单套合一个三层套合有优惠吗", "billing", "批量采购折扣", "hard"),
    ("我买的有点多有没有什么优惠", "billing", "批量采购折扣", "hard"),
    ("我前几天加入购物车了今天看失效了", "general", "订单查询", "hard"),
    ("能发顺丰吗？我上次等太久了", "general", "顺丰加急", "hard"),
    ("行退多少", "billing", "退款金额计算", "hard"),
    ("那老客户还是原价", "general", "老客回馈", "hard"),
    ("顺便问下现在有什么优惠券吗", "billing", "优惠券使用规则", "hard"),
]

SOURCE_MAP = {
    "为什么有运费": "ecd",
    "买多少斤包邮": "ecd",
    "价格可以有优惠吗": "ecd",
    "发中通圆通吗": "ecd",
    "取消重新加双袜子重新拍你说不用": "ecd",
    "我买5个单套合一个三层套合有优惠吗": "ecd",
    "我买的有点多有没有什么优惠": "ecd",
    "我前几天加入购物车了今天看失效了": "ecd",
    "能发顺丰吗？我上次等太久了": "golden_mt",
    "行退多少": "ecd",
    "那老客户还是原价": "ecd",
    "顺便问下现在有什么优惠券吗": "golden_mt",
}


def main() -> None:
    rows = list(csv.DictReader(open(GOLDEN, encoding="utf-8-sig")))
    existing = {r["query"] for r in rows}
    fieldnames = list(rows[0].keys())
    added = 0
    for query, dom, titles, diff in NEW_ANNOTATIONS:
        if query in existing:
            print(f"[skip] 已存在: {query}")
            continue
        rows.append({
            "query": query,
            "domain": dom,
            "expected_titles": titles,
            "difficulty": diff,
            "source": SOURCE_MAP.get(query, "ecd"),
            "suggest": "KEEP",
            "candidates": "",
        })
        added += 1
    with open(GOLDEN, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"追加 {added} 条，金标集现共 {len(rows)} 条")


if __name__ == "__main__":
    main()
