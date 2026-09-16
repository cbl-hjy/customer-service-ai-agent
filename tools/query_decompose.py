"""
A5 查询分解（2026-08-17）：复合查询 → 子查询拆解。

背景：mt-402「收到的手机屏幕碎了，能换货吗？能给点补偿吗？」被分类为 complaint → 升级，
但换货(general 域)与补偿标准(complaint 域)均为 KB 可答诉求，单域分类把"多诉求"压成单标签
导致误升级。A5 = 规则判定复合 + LLM 拆解子查询 + 分别检索汇总。

设计（对齐既有架构约束）：
- 复合检测是确定性规则（多问句/并列词 + 多域词共现），零 LLM 成本前置；
  仅检测"是否可能复合"，宁可漏（走原路径）不可错（拆坏单诉求）。
- LLM 拆解只在检测命中后调用，输出 JSON [{sub_query, domain}]；domain 限 5 业务域。
- fail-safe：拆解异常/非 JSON/空 → 返回 [] → 上层回退单路径（不改变原行为）。
- 拆解器由 make_graph 注入（与 A4 改写器同纪律），评估/单测不注入 = 纯检索口径；
  全局副作用必须配对 clear 原语 + 测试 fixture 隔离。
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from exceptions import LLMServiceUnavailable

logger = logging.getLogger(__name__)

# =============================================================================
# 复合检测（确定性规则，零 LLM 成本）
# =============================================================================

# 业务域关键词（与 classify_query 各域边界对齐；仅用于"多域共现"的复合线索，
# 不做细粒度意图判断——那是分类器职责）
_DOMAIN_KEYWORDS: Dict[str, tuple] = {
    "product": ("耳机", "手机", "电脑", "平板", "手表", "型号", "配置", "参数", "推荐", "哪个好", "X1", "X2", "X3"),
    "tech": ("黑屏", "闪退", "没声音", "无声", "充不进", "死机", "重启", "蓝屏", "卡顿", "连不上", "故障", "坏了"),
    "billing": ("退货", "退款", "运费", "发票", "分期", "手续费", "付款", "报销", "税号", "开票", "退款多久", "到账"),
    "complaint": ("投诉", "补偿", "赔偿", "态度", "敷衍", "索赔", "差评", "举报"),
    "general": ("发货", "物流", "快递", "赠品", "满赠", "优惠券", "地址", "换货", "补发", "预售", "库存", "缺货"),
}

# 问句分隔符：多个问句是复合的强信号
_QUESTION_SEP = re.compile(r"[?？]")

# 并列/追加连词（"还能""顺便""另外""也""还有"等）——单句内多诉求的弱信号
_CONJUNCTIONS = ("还能", "顺便", "另外", "也", "还有", "以及", "同时", "再")


def detect_compound(query: str) -> bool:
    """确定性判定查询是否可能为复合（多诉求共现）。

    信号（任一命中即判复合，宁漏不可错）：
    1. 多个问句：按 ？/? 切分出现 ≥2 个问句
    2. 单句内并列连词 + 至少 2 个不同业务域关键词共现
    """
    if not query or not query.strip():
        return False

    # 信号 1：多问句
    q_parts = [p.strip() for p in _QUESTION_SEP.split(query) if p.strip()]
    if len(q_parts) >= 2:
        return True

    # 信号 2：并列连词 + 多域共现（仅对单问句/陈述句生效）
    if any(c in query for c in _CONJUNCTIONS):
        hit_domains = set()
        for dom, words in _DOMAIN_KEYWORDS.items():
            if any(w in query for w in words):
                hit_domains.add(dom)
        if len(hit_domains) >= 2:
            return True

    return False


# =============================================================================
# LLM 拆解子查询（命中复合才调用；结构化输出，fail-safe）
# =============================================================================

_ALLOWED_DOMAINS = ("product", "tech", "billing", "complaint", "general")

_DECOMPOSE_SYSTEM_PROMPT = """你是客服查询拆解专家。客户的一句话里可能包含多个独立诉求（如"能换货吗？能给点补偿吗？"=换货+补偿两个诉求）。

请把查询拆解为**独立的子查询**，每个子查询是单一条目的独立问题，并标注其所属业务域：
- product: 产品信息（型号、配置、价格、选型）
- tech: 技术支持（故障、报错、使用问题）
- billing: 账单/支付（退款、退货、发票、分期、运费）
- complaint: 投诉/补偿（不满、补偿标准、索赔）
- general: 一般咨询（物流、发货、赠品、优惠、换货、地址）

要求：
1. 子查询必须保留原句中的实体信息（产品型号、数量、场景），不得丢词
2. 每个子查询只含一个诉求
3. 最多拆成 4 个子查询；无法拆分（单诉求）时只返回 1 个
4. 若子诉求涉及"补偿/索赔"类，归 complaint；涉及"换货/退货"归 general（退货申请归 billing）
5. 只输出一个 JSON 对象，格式：
   {"sub_queries": [{"sub_query": "能换货吗", "domain": "general"}, {"sub_query": "能给点补偿吗", "domain": "complaint"}]}
不要输出任何其他文字。JSON："""


def _normalize_domain(raw: str) -> str:
    """规范域标签（抗大小写/多余字符）。非法域 → general（fail-safe 兜底，不丢弃子查询）。"""
    if not raw:
        return "general"
    text = str(raw).strip().lower()
    for d in _ALLOWED_DOMAINS:
        if d == text:
            return d
    # 兼容长名（如 "general_inquiry" → general）
    for d, alias in (("general", "general_inquiry"), ("product", "product_info"),
                     ("tech", "technical_support")):
        if alias == text:
            return d
    return "general"


def parse_decomposed(raw: str) -> List[Dict[str, str]]:
    """解析拆解输出：优先 JSON 数组 → 规范化；异常 → 空列表（fail-safe 回退单路径）。"""
    if not raw or not str(raw).strip():
        return []
    text = str(raw).strip()
    # 兼容：直接数组 或 包在对象里 {"sub_queries": [...]}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            arr = obj.get("sub_queries") or obj.get("sub_query") or []
        else:
            arr = obj
        if not isinstance(arr, list):
            return []
        subs = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            sub = str(item.get("sub_query") or item.get("query") or "").strip()
            if not sub:
                continue
            subs.append({"sub_query": sub, "domain": _normalize_domain(item.get("domain"))})
        return subs
    except Exception as e:  # noqa: BLE001  拆解失败 = fail-safe 回退
        logger.warning("A5 拆解输出解析失败（回退单路径）: %s", e)
        return []


@tool
def decompose_query(query: str, llm=None) -> str:
    """把复合查询拆解为独立子查询（各带业务域），返回 JSON 数组字符串。

    Args:
        query: 当前用户查询（已判定为可能复合）
        llm: LLM 实例
    """
    messages = [
        SystemMessage(content=_DECOMPOSE_SYSTEM_PROMPT),
        HumanMessage(content=f"请拆解以下查询：{query}"),
    ]
    try:
        response = llm.invoke(messages, response_format={"type": "json_object"})
        result = (getattr(response, "content", "") or "").strip()
        subs = parse_decomposed(result)
        if not subs:
            # 非 JSON 异常 → 尝试字符串兜底（单个子查询 = 原句）
            return json.dumps([{"sub_query": query, "domain": "general"}], ensure_ascii=False)
        return json.dumps(subs, ensure_ascii=False)
    except LLMServiceUnavailable:
        raise  # 通道故障：交给上层降级（与分类器同协议）
    except Exception as e:
        logger.warning(f"查询拆解失败: {e}")
        return json.dumps([{"sub_query": query, "domain": "general"}], ensure_ascii=False)


# =============================================================================
# 拆解器注入（与 A4 改写器同纪律：make_graph 注入，评估/单测不注入）
# =============================================================================

_decomposer = None


def set_query_decomposer(fn) -> None:
    """注入查询拆解器（应用层启动时调用）。fn(query) -> JSON 字符串或空。"""
    global _decomposer
    _decomposer = fn


def clear_query_decomposer() -> None:
    """清除拆解器（评估/测试隔离用）。"""
    global _decomposer
    _decomposer = None


def get_query_decomposer():
    """返回已注入的拆解器（None = 未注入，纯检索口径）。"""
    return _decomposer
