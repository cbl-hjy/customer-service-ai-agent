"""A5 复合查询（2026-08-17）：多诉求查询拆解 + 分别检索综合回答。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

import logging
from datetime import datetime

from exceptions import LLMServiceUnavailable
from llm import get_llm
from llm.stream_sink import call_llm  # P1 流式：综合回答正文逐 token 推送
from citations import _attach_compound_citations
from .dialogue import _compact_after_append
from .escalate import escalate_node

logger = logging.getLogger(__name__)


def _try_decompose(query: str) -> list:
    """复合查询拆解：detect_compound 命中且拆解出 ≥2 子查询时返回列表，否则 []。

    fail-safe：拆解器未注入 / 检测未命中 / 拆解异常 / 子查询数 <2 → 返回 []
    （调用方走原路径，行为零变化）。通道故障（熔断）由拆解器抛 LLMServiceUnavailable，
    在此捕获并按 [] 处理——复合拆解是增强不是必需，故障时保原行为最安全。
    """
    if not query:
        return []
    try:
        from tools.query_decompose import detect_compound as _detect, get_query_decomposer, parse_decomposed
        if not _detect(query):
            return []
        fn = get_query_decomposer()
        if fn is None:
            return []
        raw = fn(query)
        subs = parse_decomposed(raw)
        return subs if len(subs) >= 2 else []
    except LLMServiceUnavailable:
        logger.warning("A5 复合拆解通道故障，回退单路径: query=%s", query[:30])
        return []
    except Exception as e:  # noqa: BLE001  拆解失败不阻断主流程
        logger.warning("A5 复合拆解失败（回退单路径）: %s", e)
        return []


def _compound_retrieve_all(subs: list) -> list:
    """子查询分别检索（与 agent 同判定路径）。返回 [(sub, domain, kb_text, kb_titles)]，
    任一 miss 返回 []。kb_titles 从检索文本解析（与注入内容一致，引用溯源真实）。"""
    from kb_retriever import retrieve as _kb_retrieve
    import re as _re
    hits = []
    for s in subs:
        text = _kb_retrieve(s.get("domain", "general"), s.get("sub_query", ""))
        if not text or not text.strip():
            return []  # 任一子诉求无 KB 依据 → 整体升级兜底（不编造未覆盖诉求）
        # 解析检索文本中的条目标题（格式：【category】\n• title：content）
        titles = [t for t in _re.findall(r"•\s*(.+?)[：:]", text) if t and t.strip()]
        hits.append((s["sub_query"], s["domain"], text, titles))
    return hits


# 复合回答的生成约束（V6 事实边界同源：只陈述 KB 明确内容，不编造时效/金额）
_COMPOUND_SYSTEM_PROMPT = """你是星环数码智能客服。用户的一句话中包含多个独立诉求，请基于【知识库内容】逐条完整回答每一个诉求。

要求：
1. 逐条回应每个子诉求，不遗漏任何一项；每个诉求的答案都明确对应（可用"关于您问的XXX"分段）
2. 只陈述知识库明确写出的规则、时效、金额与条件；知识库未明确规定的不得自行补充、推断或编造
3. 不得改述知识库中的关键限定词（如把"下单前"改为"发货前"）
4. 回答自然、友好、专业"""


def compound_node(state: dict) -> dict:
    """复合查询处理节点：子查询分别检索 → 全部命中 → LLM 综合回答；任一 miss → 升级兜底。

    多轮上下文在 state.persisted_dialogue 中（case 内轮次保持），综合回答基于当前 query
    的子诉求拆分；引用溯源由 _attach_compound_citations 附加（合并各子查询命中标题）。
    """
    subs = state.get("compound_subs") or []
    if len(subs) < 2:
        # 理论上不会到这（classify 已过滤），防御性回退：走升级兜底
        return _compound_escalate(state, "复合拆解结果不足，转人工核实")

    hits = _compound_retrieve_all(subs)
    if not hits:
        return _compound_escalate(state, "复合诉求中部分内容知识库未覆盖，转人工核实")

    # 拼接 KB 上下文 + 子诉求清单
    kb_block = "\n\n".join(
        f"【子诉求 {i + 1}：{sub}】（{domain}域）\n{text}" for i, (sub, domain, text, _titles) in enumerate(hits)
    )
    user_content = (
        f"用户查询：{state.get('customer_query', '')}\n\n"
        f"拆解出的子诉求与知识库内容：\n{kb_block}\n\n"
        "请逐条回答全部子诉求。"
    )
    try:
        llm = get_llm()
        from langchain_core.messages import HumanMessage, SystemMessage
        resp = call_llm(llm, [SystemMessage(content=_COMPOUND_SYSTEM_PROMPT), HumanMessage(content=user_content)])
        response = getattr(resp, "content", "") or ""
    except Exception as e:  # noqa: BLE001  LLM 失败 → 升级兜底（不编造）
        logger.warning("A5 复合回答 LLM 失败，升级兜底: %s", e)
        return _compound_escalate(state, f"复合回答生成失败: {str(e)[:80]}")

    state["response"] = response
    state["current_agent"] = "综合客服"
    state["tools_used"].append("compound_processing")

    # A5 引用溯源：合并各子查询命中标题（与注入内容一致，judge 基座真实可核）
    _attach_compound_citations(state, hits)

    # 助手轮次写入 checkpointer 状态（与 agent 节点一致）
    _compact_after_append(state, {
        "content": str(state["response"]),
        "is_user": False,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    return state


def _compound_escalate(state: dict, reason: str) -> dict:
    """复合查询升级兜底（任一子诉求 KB 未覆盖 / LLM 失败）：走既有升级路径，不编造。"""
    state["escalated"] = True
    state["escalation_reason"] = reason
    state["compound_subs"] = []
    return escalate_node(state)
