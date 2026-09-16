"""升级节点（C4）：fail-safe 出口——生成升级摘要 + 标记人工，不硬答、不编造。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

import json
import logging
from datetime import datetime

from .dialogue import _compact_after_append

logger = logging.getLogger(__name__)


def escalate_node(state: dict) -> dict:
    """升级节点：低置信/复杂/投诉工单转人工，生成结构化摘要（终结性节点，不循环）。
    若已升级过（escalation_summary 非空），则本次为追问，走安抚回复，不重复升级。
    2026-08-16 A4 增强：升级态追问按原始升级域检索一次，KB 可答则安抚+关键信息
    （mt-308"你们的补偿标准是什么"——补偿标准条目在 complaint 域，追问分类摇摆
    时两条路径均检索不可达；按 escalation_summary 记录的原域检索可稳定命中。
    无关追问由检索双低门禁兜底，不会误答）。
    """
    if state.get("escalation_summary"):
        # 已升级过，本次追问：安抚回复（+ 原域 KB 信息补全），不重复升级
        state["current_agent"] = "智能客服"
        response = (
            "您的问题已转人工客服处理，请耐心等待客服与您联系。"
            "如有其他问题，可继续留言，我们会尽快反馈。"
        )
        try:
            from kb_retriever import retrieve as _kb_retrieve
            _label2domain = {
                "complaint": "complaint", "billing": "billing",
                "technical_support": "tech", "product_info": "product",
                "general_inquiry": "general",
            }
            _orig = json.loads(state["escalation_summary"])
            _dom = _label2domain.get(_orig.get("label", ""), "general")
            _hit = _kb_retrieve(_dom, state.get("customer_query", ""))
            if _hit:
                _first = _hit.split("\n")[0].lstrip("•【】· ").strip()
                if _first:
                    response += f"\n关于您追问的问题，可先参考：{_first}"
                    state["tools_used"].append("escalation_followup_kb")
        except Exception as e:  # noqa: BLE001  补全失败退回纯安抚
            logger.warning("升级态追问 KB 补全失败（退回纯安抚）: %s", e)
        state["response"] = response
        # 写入持久化对话（V12：统一接入点触发压缩）
        _compact_after_append(state, {
            "content": str(state["response"]),
            "is_user": False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["tools_used"].append("escalation_comfort")
        return state

    # 首次升级
    summary = {
        "session_id": state.get("session_id", ""),
        "customer_query": state.get("customer_query", ""),
        "label": state.get("query_type", ""),
        "confidence": state.get("query_confidence", 0.0),
        "complexity": state.get("query_complexity", "medium"),
        "reason": state.get("escalation_reason", ""),
    }
    state["escalation_summary"] = json.dumps(summary, ensure_ascii=False)
    state["escalated"] = True
    state["current_agent"] = "人工客服"
    reason = state.get("escalation_reason", "")
    state["response"] = (
        f"您的工单已升级至人工客服处理（原因：{reason}）。"
        "人工客服将尽快跟进，请您保持关注。"
    )
    # 写入持久化对话（与业务 agent 节点一致，V12：统一接入点触发压缩）
    _compact_after_append(state, {
        "content": str(state["response"]),
        "is_user": False,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    state["tools_used"].append("escalation")
    return state
