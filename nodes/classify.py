"""查询分类节点（C2 结构化分类 + C4 升级决策 + A5 复合触发）。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

import json
import logging
from datetime import datetime

from langgraph.config import get_config

from exceptions import LLMServiceUnavailable
from llm import get_llm
from tools import classify_query
from routing import should_escalate, wants_human
from .state import OUT_OF_SCOPE_REPLY
from .dialogue import _compact_after_append
from .compound import _try_decompose

logger = logging.getLogger(__name__)


def classify_query_node(state: dict) -> dict:
    """Classify customer query"""
    try:
        cfg = get_config()
        tid = (cfg.get("configurable") or {}).get("thread_id")
        if tid:
            state["session_id"] = str(tid)
    except RuntimeError:
        pass

    # 初始化状态对象
    if "session_id" not in state or not state.get("session_id"):
        import uuid
        state["session_id"] = str(uuid.uuid4())

    if "tools_used" not in state:
        state["tools_used"] = []

    if "conversation_history" not in state:
        state["conversation_history"] = []

    if "persisted_dialogue" not in state or state.get("persisted_dialogue") is None:
        state["persisted_dialogue"] = []

    if "dialogue_summary" not in state or state.get("dialogue_summary") is None:
        state["dialogue_summary"] = ""

    if "next_agent" not in state:
        state["next_agent"] = ""

    if "messages" not in state:
        state["messages"] = []

    if "query_confidence" not in state:
        state["query_confidence"] = 0.0

    if "query_complexity" not in state:
        state["query_complexity"] = "medium"

    if "escalated" not in state:
        state["escalated"] = False

    if "escalation_reason" not in state:
        state["escalation_reason"] = ""

    if "escalation_summary" not in state:
        state["escalation_summary"] = ""

    # 获取必需字段
    customer_query = state.get("customer_query", "")
    session_id = state["session_id"]

    if not customer_query:
        state["response"] = "Error: No customer query provided"
        state["query_type"] = "general_inquiry"
        return state

    # 使用分类工具
    try:
        llm_instance = get_llm()
        # 使用正确的工具调用方式（C2：classify_query 返回 JSON，含 label/confidence/complexity）
        try:
            # W4 修复：多轮上下文信号，解决跨轮指代（"那款/这个/它"）分类抖动。
            # 取上一轮用户消息（persisted_dialogue 最后一条 is_user）等构造信号，注入分类器。
            context_signal = None
            pd_hist = list(state.get("persisted_dialogue") or [])
            prev_user = next(
                (m.get("content", "") for m in reversed(pd_hist) if m.get("is_user")),
                "",
            )
            if prev_user:
                context_signal = f"上一轮用户提到：{prev_user}"
            result = classify_query.invoke(
                {"query": customer_query, "llm": llm_instance, "context_signal": context_signal}
            )
            data = json.loads(result) if isinstance(result, str) else result
            query_type = data.get("label", "general_inquiry")
            state["query_confidence"] = float(data.get("confidence", 0.0))
            state["query_complexity"] = str(data.get("complexity", "medium"))
            state["llm_unavailable"] = False  # 通道正常，清残留标记（state 跨轮复用）
        except LLMServiceUnavailable as e:
            # 【韧性】通道故障（403/熔断/5xx）：不是"用户问题不确定"，而是"系统暂时不可用"。
            # 走友好降级提示，不升级人工（升级了人工也调不了 LLM，徒增客服负担）。
            logger.warning(f"LLM 通道不可用，走降级提示: {e}")
            state["query_type"] = "general_inquiry"
            state["query_confidence"] = 0.0
            state["query_complexity"] = "medium"
            state["escalated"] = False
            state["escalation_reason"] = ""
            state["escalation_summary"] = ""
            state["llm_unavailable"] = True
            state["current_agent"] = "智能客服"  # final_response_node 依赖该字段
            state["response"] = (
                "抱歉，智能客服服务暂时不可用（可能是服务限流或额度已用尽），"
                "请稍候片刻再试。若情况紧急，可拨打人工客服热线。"
            )
            return state
        except Exception as e:
            logger.error(f"工具调用失败: {e}")
            # 回退到基础分类逻辑（分类解析失败，非通道故障 → 保守升级兜底）
            query_type = "general_inquiry"
            state["query_confidence"] = 0.0
            state["query_complexity"] = "medium"
            state["llm_unavailable"] = False  # 非通道故障，确保不误跳 final_response
    except Exception as e:
        logger.error(f"查询分类失败: {e}")
        query_type = "general_inquiry"
        state["query_confidence"] = 0.0
        state["query_complexity"] = "medium"
        state["llm_unavailable"] = False  # 非通道故障，确保不误跳 final_response

    # 更新状态
    state["query_type"] = query_type
    state["tools_used"].append("query_classification")

    # 写入由 checkpointer 持久化的对话（用户轮次，V12：统一接入点触发压缩）
    _compact_after_append(state, {
        "content": str(customer_query),
        "is_user": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # V13：移除双写——会话权威记录是 checkpointer persisted_dialogue（web 路径全部走它），
    # session_manager 仅作 base_agent state 缺失兜底，不再冗余写入（原 5 处 add_message 全移除）

    # v15 修复：用户主动求人工 → 强制升级（代码硬边界，优先于分类/护栏）
    # 用户明确要求转人工是最强升级信号，LLM 无感知、不碰 prompt
    if wants_human(customer_query):
        state["escalated"] = True
        state["escalation_reason"] = "用户主动要求转人工"
        return state

    # 护栏：超出范围直接固定回复，不进入业务智能体
    if state["query_type"] == "out_of_scope":
        state["response"] = OUT_OF_SCOPE_REPLY
        state["current_agent"] = "智能客服"
        # V12：统一接入点触发压缩
        _compact_after_append(state, {
            "content": OUT_OF_SCOPE_REPLY,
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["tools_used"].append("out_of_scope_refusal")
        return state

    # A5 复合查询（2026-08-17）：多诉求查询 → 拆解子查询 → 分别检索综合回答。
    # 触发策略（回归风险最小）：仅当"原路径会升级"时才尝试复合分解——
    #   mt-402 换货+补偿被压成 complaint → 原路径误升级；拆解后两子诉求 KB 均可答，
    #   改由 compound 节点综合回答。其余已通过 case（原路径不升级）行为零变化。
    # 拆解失败/子查询检索 miss → compound 节点内升级兜底（不编造，fail-safe）。
    state["compound_subs"] = []
    escalate, reason = should_escalate(
        state["query_type"],
        state.get("query_confidence", 0.0),
        state.get("query_complexity", "medium"),
    )
    if escalate and not state.get("escalation_summary"):
        state["compound_subs"] = _try_decompose(customer_query)
        if state["compound_subs"]:
            escalate, reason = False, ""
    # W4 修复：已升级会话的追问不重复升级。
    # 之前 escalated=True 会在 route_after_classify 无条件走 escalate，导致升级后用户追问
    # （如"多久能处理"）反复复用升级文案，追问永不接住（mt-003 稳定缺陷）。
    # escalation_summary 仅由 escalate_node 首次升级时写入、跨轮保留，作为"已升级过"标记。
    if escalate and state.get("escalation_summary"):
        # 已升级过：本轮是追问/新反馈 → 交给对应业务 agent 直接回答，不再强制升级
        escalate = False
        reason = ""
    state["escalated"] = escalate
    state["escalation_reason"] = reason

    return state
