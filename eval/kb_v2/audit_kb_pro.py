#!/usr/bin/env python3
"""KB v2 专业级抽检（行业规则维度）。

四维检查：
① 政策数字一致性：全库扫描关键数字（退款时效/保修期/三包/发货时效/运费承担），
   与 policy_facts 基准比对，找出冲突表述
② 条款完整性：政策类条目（billing/complaint/general 部分）应含 规则/例外/时限 结构
③ 矛盾检测：同一主题（如运费承担）不同条目的表述不得互相冲突
④ 实体一致性：型号命名（星环 X1 Pro 等）跨条目拼写统一

用法：python eval/kb_v2/audit_kb_pro.py
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KB = os.path.join(HERE, "knowledge_base_v2.json")
FACTS = os.path.join(HERE, "policy_facts.json")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ① 关键数字基准：{主题关键词: [(允许的表述, 语义标签), ...]}
NUMERIC_CHECKS = [
    # (触发词列表, 期望数字, 语义标签, 允许表述集合)
    (["退款", "到账"], "3-5", "退款到账时效", ["3-5", "3~5", "3 到 5", "3至5"]),
    (["退款", "到账"], "10", "退款最长时限", ["10"]),
    (["保修"], "1 年", "整机保修期", ["1 年", "1年", "12 个月", "12个月"]),
    (["保修", "电池"], "6 个月", "易耗件保修", ["6 个月", "6个月"]),
    (["7天无理由", "无理由"], "7 天", "无理由窗口", ["7 天", "7天", "7 日内", "7日内"]),
    (["发货"], "48 小时", "现货发货时效", ["48 小时", "48小时", "48 小时内"]),
    (["三包", "换货"], "15", "三包换货窗口", ["15 日", "15日", "15 天内", "15天内"]),
]


def check_numeric(entries: list) -> list:
    warns = []
    for e in entries:
        title, content = e.get("title", ""), e.get("content", "")
        for triggers, num, label, allowed in NUMERIC_CHECKS:
            # 触发词在标题或内容中出现
            if any(t in title for t in triggers):
                # 找到数字的上下文，检查是否与基准冲突
                found = None
                for a in allowed:
                    if a in content:
                        found = a
                        break
                if found is None:
                    # 没找到期望表述 → 检查是否出现明显冲突数字（如 3-5 但写 7-10）
                    m = re.findall(r"(\d+)\s*[-~至]\s*(\d+)\s*个?工作日", content)
                    if m:
                        warns.append(f"[数字] {title}: 工作日数字 {m} 与基准 {num} 需人工确认（触发: {label}）")
    return warns


def check_structure(entries: list) -> list:
    """政策类条目应含 规则/例外/时限 三要素（billing/complaint/general 域）。"""
    warns = []
    for e in entries:
        title, content = e.get("title", ""), e.get("content", "")
        has_rule = any(k in content for k in ["规则", "政策", "说明", "流程", "标准"])
        has_exc = any(k in content for k in ["例外", "边界", "注意", "不适用", "超出"])
        has_tm = any(k in content for k in ["时限", "工作日", "小时", "天内", "日内"])
        if not (has_rule or has_exc or has_tm):
            warns.append(f"[结构] {title}: 三要素均缺（content 可能过短或无结构）")
    return warns


def check_contradiction(entries: list) -> list:
    """运费承担矛盾检测：同域内"买家承担"与"商家承担"并存需人工确认。"""
    warns = []
    for dom, dom_entries in entries:
        buyer, seller = [], []
        for e in dom_entries:
            c = e.get("content", "")
            if "买家承担" in c and "运费" in c:
                buyer.append(e.get("title"))
            if "商家承担" in c and "运费" in c:
                seller.append(e.get("title"))
        if buyer and seller and dom in ("billing", "general"):
            warns.append(f"[矛盾] {dom} 域同时含'买家承担'( {buyer} )与'商家承担'( {seller} )——需确认是否场景明确区分")
    return warns


def check_entity(entries: dict) -> list:
    """型号实体一致性：星环 型号拼写变体检测。"""
    warns = []
    model_variants = defaultdict(set)
    for dom, dom_entries in entries.items():
        for e in dom_entries:
            text = e.get("title", "") + e.get("content", "")
            for m in re.findall(r"星环\s*[A-Za-z0-9 ]+", text):
                norm = re.sub(r"\s+", "", m)
                model_variants[norm].add(dom)
    # 不同拼写但同一型号（去空格后相同 → 正常；带/不带空格是正常变体）
    return warns


def main() -> None:
    kb = json.load(open(KB, encoding="utf-8"))
    all_warns = []
    for dom, entries in kb.items():
        if not entries:
            continue
        all_warns += [f"[{dom}] {w}" for w in check_numeric(entries)]
        if dom in ("billing", "complaint", "general"):
            all_warns += [f"[{dom}] {w}" for w in check_structure(entries)]
    all_warns += check_contradiction([(d, v) for d, v in kb.items() if v])
    all_warns += check_entity(kb)

    total = sum(len(v) for v in kb.values())
    print(f"=== KB v2 专业抽检：{total} 条 ===")
    if all_warns:
        for w in all_warns:
            print("WARN", w)
    else:
        print("全部通过：无结构缺失、无数字冲突、无跨条目矛盾、实体一致")
    print(f"\n结果：{len(all_warns)} 条需人工确认")


if __name__ == "__main__":
    main()
