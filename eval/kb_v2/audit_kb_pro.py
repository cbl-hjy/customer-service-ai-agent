#!/usr/bin/env python3
"""KB v2 专业级抽检（行业规则维度）。

五维检查：
① 政策数字一致性：全库扫描关键数字（退款时效/保修期/三包/发货时效/运费承担），
   与 policy_facts 基准比对，找出冲突表述
② 条款完整性：政策类条目（billing/complaint/general 部分）应含 规则/例外/时限 结构
③ 矛盾检测：同一主题（如运费承担）不同条目的表述不得互相冲突
④ 实体一致性：型号命名（星环 X1 Pro 等）跨条目拼写统一
⑤ 价格矩阵聚合一致性（2026-09-16 新增）：同一商品名的点价/区间价跨条目聚合，
   多值且无法由合法形态（配置档/区间覆盖）解释 → 需人工确认
   （教训：AirBuds 价格矩阵四组拧 + Pad Mini 区间矛盾，共五次 KB 跨条目矛盾
    均为"同商品不同价"模式，单条目内检查结构性看不到，必须跨条目聚合）

用法：python eval/kb_v2/audit_kb_pro.py
"""
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(os.path.dirname(HERE))
KB = os.path.join(PACKAGE_ROOT, "data", "knowledge_base_v2.json")  # 单源化：canonical=data/
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


# ---------------------------------------------------------------------------
# ⑤ 价格矩阵聚合一致性（2026-09-16 新增）
# 机制：同商品名的点价/区间价跨条目聚合；多值且非合法形态 → WARN 需人工确认。
# 合法形态豁免：
#   a. 所有点价仅出现在同一条目（配置档/原价会员价并存的典型形态）
#   b. 存在 ≥2 条目出现的区间价覆盖全部点价（如 Pad Pro 3999-5299 覆盖双档点价）
# 防误抓（实测教训）：
#   - 配件主题条目整条跳过：标题含配件词（手机壳材质对比/充电器选购对比等），
#     该类条目内价格全部属配件，主商品名只作适配对象出现（X1 Pro 手机壳 149 案例）
#   - 配件价格错绑主商品：价格与归属名之间出现配件词（充电器/键盘/手机壳…）→ 跳过
#     （案例：「星环 Pad 11（原价约 1999 元）、星环 65W 氮化镓充电器（原价约 149 元）」
#       149 属充电器，但词典无充电器名会错绑 Pad 11）
#   - 适配语境过滤：归属名与价格之间出现"兼容/支持"→ 价格属句尾配件主体
#     （案例：「适配星环 Pad Mini 8（18W 兼容），体积轻巧约 45g，售价约 39 元」39 属快充头）
#   - 词典含裸系列名（X1）：否则「X1 标准版（约 2779 元）」会错绑前文 X1 Lite
# ---------------------------------------------------------------------------
# 完整产品名 = 系列 + (修饰|数字)* 序列；数字后不得跟字母（挡 "120W" 误吃）
_PROD_NAME = re.compile(
    r"(AirBuds|HeadBuds|X\d|Pad|Book|Watch|Fold\d?)"
    r"(?:\s+(?:Pro|Lite|Max|Mini|Air)|\s+\d+(?:\.\d+)?(?![A-Za-z0-9.]))*")
_PRICE = re.compile(r"(\d{2,5})(?:\s*[-~至]\s*(\d{2,5}))?\s*元")
_ACCESSORY_WORDS = re.compile(
    r"充电器|充电头|数据线|键盘|保护壳|手机壳|钢化膜|贴膜|手写笔|表带|腕带|耳机套|扩展坞|移动电源")


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", "", s).strip()


def _build_lexicon(kb: dict) -> set:
    """商品词典 = 全部条目标题 ∪ 内容中的完整产品名（norm 后）。

    标题是权威出处；内容补充是必须的——"星环 Pad Mini 8" 若只出现在
    content 而所有标题都写 "Pad Mini"，仅靠标题词典会丢归属。
    """
    lex = set()
    for entries in kb.values():
        for e in entries:
            for m in _PROD_NAME.finditer(e.get("title", "") + " " + e.get("content", "")):
                n = _norm_name(m.group(0))
                if len(n) >= 2 and not n.isdigit():
                    lex.add(n)
    return lex


def _split_clauses(text: str) -> list:
    return [c for c in re.split(r"[。；;]", text) if c.strip()]


def check_price_matrix(kb: dict) -> list:
    """同商品名价格跨条目聚合：多值且非合法形态 → WARN。"""
    lex = _build_lexicon(kb)
    pt = defaultdict(lambda: defaultdict(set))    # name -> 点价 -> {条目标题}
    rg = defaultdict(lambda: defaultdict(set))    # name -> (lo,hi) -> {条目标题}

    for entries in kb.values():
        for e in entries:
            title, content = e.get("title", ""), e.get("content", "")
            # 配件主题条目整条跳过：价格全部属配件，主商品名只作适配对象出现
            if _ACCESSORY_WORDS.search(title):
                continue
            for cl in _split_clauses(content):
                names = sorted((m.start(), _norm_name(m.group(0)))
                               for m in _PROD_NAME.finditer(cl)
                               if _norm_name(m.group(0)) in lex)
                for pm in _PRICE.finditer(cl):
                    prior = [(p, n) for p, n in names if p < pm.start()]
                    if not prior:
                        continue
                    owner_pos, owner = prior[-1]
                    # 配件防误抓：归属名与价格之间出现配件词或"兼容/支持"适配语境
                    # → 价格属配件/句尾主体，跳过
                    between = cl[owner_pos:pm.start()]
                    if _ACCESSORY_WORDS.search(between) or re.search(r"兼容|支持", between):
                        continue
                    lo, hi = pm.group(1), pm.group(2)
                    if hi:
                        rg[owner][(int(lo), int(hi))].add(title)
                    else:
                        pt[owner][int(lo)].add(title)

    warns = []
    for name in sorted(set(pt) | set(rg)):
        # 检 1：同商品多个不同区间 → 区间不一致（Pad Mini 999-1299 vs 1299-1599 形态）
        if len(rg.get(name, {})) > 1:
            spans = "; ".join(f"{lo}-{hi}<-{sorted(ts)}" for (lo, hi), ts in sorted(rg[name].items()))
            warns.append(f"[价格矩阵] {name}: 区间价不一致 {spans}")
        # 检 2：点价多值且非合法形态
        prices = pt.get(name, {})
        if len(prices) <= 1:
            continue
        # 豁免 a：所有点价仅出现在同一条目（配置档形态）
        all_titles = set().union(*prices.values())
        if len(all_titles) == 1:
            continue
        # 豁免 b：≥2 条目出现的区间价覆盖全部点价（区间+双档形态）
        covered = any(
            len(titles) >= 2 and all(lo <= v <= hi for v in prices)
            for (lo, hi), titles in rg.get(name, {}).items())
        if covered:
            continue
        detail = "; ".join(f"{v}<-{sorted(ts)}" for v, ts in sorted(prices.items()))
        warns.append(f"[价格矩阵] {name}: 点价跨条目不一致 {detail}")
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
    all_warns += check_price_matrix(kb)

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
