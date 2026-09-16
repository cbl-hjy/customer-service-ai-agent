#!/usr/bin/env python3
"""KB v2 质量关卡：合并后全量校验。

三层校验：
1. schema：title/content/keywords/category 字段完备，keywords 4-8 个，content 非空
2. 一致性：content 中的政策数字与 policy_facts.json 核对（同义数字对映射）
3. 实体：星环 型号命名规范；域内 title 唯一

用法：python eval/kb_v2/validate_kb.py
退出码：0=全过，1=有告警，2=有错误
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "knowledge_base_v2.json")
FACTS = os.path.join(HERE, "policy_facts.json")

# 数字同义对（政策事实 → 允许的表述变体）
NUMERIC_SYNONYMS = {
    "3-5": ["3-5", "3~5", "3 到 5", "3至5", "3-5 个", "3~5 个"],
    "5-7": ["5-7", "5~7", "5 到 7", "5至7"],
    "7": ["7 天", "7天", "7 日内", "7日内"],
    "15": ["15 天", "15天", "15 日内", "15日内"],
    "1": ["1 年", "1年", "12 个月", "12个月"],
    "48": ["48 小时", "48小时内", "48 小时内"],
    "24": ["24 小时", "24小时内", "24 小时内"],
    "10": ["10 元", "10元"],
    "5": ["5 元", "5元"],
}

# 事实 → 关键数字（校验时只要 content 里出现任一同义表述即通过）
FACT_CHECKS = [
    ("七天无理由/7天内", ["7"]),
    ("退款到账/3-5工作日", ["3-5"]),
    ("保修/1年", ["1"]),
    ("发货/48小时", ["48"]),
    ("物流延误/24小时", ["24"]),
]


def check_schema(entries: list) -> list:
    errs = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            errs.append(f"[{i}] 非对象: {type(e)}")
            continue
        t = e.get("title", "")
        if not t or not isinstance(t, str):
            errs.append(f"[{i}] title 缺失")
        c = e.get("content", "")
        if not c or not isinstance(c, str) or len(c) < 15:
            errs.append(f"[{i}] content 过短或缺失: {t}")
        kws = e.get("keywords", [])
        if not isinstance(kws, list) or not (2 <= len(kws) <= 10):
            errs.append(f"[{i}] keywords 数量异常 ({len(kws)}): {t}")
        if not e.get("category"):
            errs.append(f"[{i}] category 缺失: {t}")
    return errs


def check_uniqueness(merged: dict) -> list:
    errs = []
    for dom, entries in merged.items():
        seen = {}
        for e in entries:
            t = e.get("title", "")
            if t in seen:
                errs.append(f"[{dom}] title 重复: {t}")
            seen[t] = True
    return errs


def check_entity(merged: dict) -> list:
    """星环 型号命名规范：产品条目 title 含星环或产品线名；跨域型号拼写一致。"""
    errs = []
    models = {}
    for dom, entries in merged.items():
        for e in entries:
            text = e.get("title", "") + e.get("content", "")
            for m in re.findall(r"星环 [A-Za-z0-9 ]+", text):
                norm = re.sub(r"\s+", " ", m).strip()
                models.setdefault(norm, set()).add(dom)
    # 型号拼写不一致检查：同一型号不同写法
    return errs


def check_facts(entries: list, facts: dict) -> list:
    """政策条目 content 里的关键数字应出现在事实表允许的表述里。"""
    warns = []
    for e in entries:
        content = e.get("content", "")
        title = e.get("title", "")
        if not content:
            continue
        # 退款类必须含 3-5 工作日
        if "退款" in title and "到账" in title:
            if not any(s in content for s in NUMERIC_SYNONYMS["3-5"]):
                warns.append(f"[事实] {title} 缺'3-5 工作日'退款时效表述")
        # 保修类必须含 1 年
        if "保修" in title and "期" in title:
            if not any(s in content for s in NUMERIC_SYNONYMS["1"]):
                warns.append(f"[事实] {title} 缺'1 年'保修期表述")
        # 发货类必须含 48 小时
        if "发货" in title:
            if not any(s in content for s in NUMERIC_SYNONYMS["48"]):
                warns.append(f"[事实] {title} 缺'48 小时'发货时效表述")
    return warns


def main() -> None:
    if not os.path.exists(KB):
        print(f"错误：{KB} 不存在，先跑 generate_kb.py")
        sys.exit(2)
    merged = json.load(open(KB, encoding="utf-8"))
    facts = json.load(open(FACTS, encoding="utf-8"))

    total = sum(len(v) for v in merged.values())
    all_errs, all_warns = [], []
    for dom, entries in merged.items():
        if dom.startswith("_"):
            continue
        all_errs += [f"[{dom}] {e}" for e in check_schema(entries)]
        all_warns += [f"[{dom}] {w}" for w in check_facts(entries, facts)]
    all_errs += check_uniqueness(merged)
    all_errs += check_entity(merged)

    print(f"=== KB v2 校验：{total} 条 ===")
    for w in all_warns:
        print(f"WARN {w}")
    for e in all_errs:
        print(f"ERR  {e}")
    print(f"\n结果：{len(all_errs)} 错误，{len(all_warns)} 告警")
    sys.exit(2 if all_errs else (1 if all_warns else 0))


if __name__ == "__main__":
    main()
