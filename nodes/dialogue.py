"""V12 对话压缩（2026-08-14）：滚动摘要 + 定长 cap + 归档表。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

import logging
import os
from typing import Any, Dict, Union

from config import DIALOGUE_CAP, DIALOGUE_TAIL_KEEP

logger = logging.getLogger(__name__)


# 摘要上限字符数（防止摘要自身无限膨胀；被归档轮次每次最多浓缩约
# DIALOGUE_CAP - DIALOGUE_TAIL_KEEP 条用户消息，超长截断）
_DIALOGUE_SUMMARY_MAX_CHARS = int(os.environ.get("DIALOGUE_SUMMARY_MAX_CHARS", "500"))


def _compact_dialogue_if_needed(state: Union[Dict[str, Any], "AgentState"]) -> None:  # noqa: F821
    """persisted_dialogue 超 DIALOGUE_CAP 时：归档最旧轮次 + 更新滚动摘要。

    规则：
    - 保留尾部 DIALOGUE_TAIL_KEEP 条原文（prompt 注入取尾部 12 条，留余量）
    - 被归档轮次的完整原文转存 dialogue_archive 表（web 渲染合并，不丢历史）
    - 被归档轮次的用户消息浓缩为一句摘要，覆盖写入 state.dialogue_summary
      （截断到 _DIALOGUE_SUMMARY_MAX_CHARS，防摘要自身膨胀）
    - 归档失败（DB 异常）时跳过压缩——宁可保留原文也不丢历史（fail-safe）
    """
    pd = state.get("persisted_dialogue") or []
    if len(pd) <= DIALOGUE_CAP:
        return
    thread_id = state.get("session_id", "")
    if not thread_id:
        logger.warning("对话压缩跳过：无 session_id，无法归档")
        return
    archived = pd[: len(pd) - DIALOGUE_TAIL_KEEP]
    tail = pd[len(pd) - DIALOGUE_TAIL_KEEP:]
    try:
        from dialogue_archive import append_entries
        append_entries(thread_id, archived)
    except Exception as e:
        logger.warning(f"对话归档失败，跳过压缩（保留原文）: {e}")
        return
    # 滚动摘要：被归档轮次的用户消息浓缩（客服场景用户诉求是主线；AI 回复随原文归档，
    # 不进摘要——prompt 只需知道"用户问过什么"）
    user_msgs = [m.get("content", "") for m in archived if m.get("is_user")]
    summary = "；".join(str(m).replace("\n", " ").strip()[:30] for m in user_msgs if str(m).strip())
    if len(summary) > _DIALOGUE_SUMMARY_MAX_CHARS:
        summary = summary[:_DIALOGUE_SUMMARY_MAX_CHARS] + "…"
    state["dialogue_summary"] = summary
    state["persisted_dialogue"] = tail
    logger.info(
        f"V12 对话压缩: thread={thread_id[:16]}… 归档 {len(archived)} 条 → 保留 {len(tail)} 条"
    )


def _compact_after_append(state: Union[Dict[str, Any], "AgentState"], appended: dict) -> None:  # noqa: F821
    """append 一条轮次后触发压缩检查（统一接入点：5 个 append 点共用）。"""
    pd = state.get("persisted_dialogue") or []
    pd.append(appended)
    state["persisted_dialogue"] = pd
    _compact_dialogue_if_needed(state)
