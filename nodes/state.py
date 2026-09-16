"""AgentState 状态定义 + 越界固定回复。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

from typing import Any, List, TypedDict


# 超出客服范围时的固定回复（护栏：不调用业务智能体）
OUT_OF_SCOPE_REPLY = (
    "抱歉，这里是智能客服，仅处理与产品、技术、账单、投诉及相关售后政策类问题；"
    "请用一句话说明您的具体业务诉求，我很乐意协助。"
)


# 定义状态类型
class AgentState(TypedDict):
    session_id: str
    messages: List[Any]
    current_agent: str
    customer_query: str
    query_type: str
    query_confidence: float
    query_complexity: str
    escalated: bool
    escalation_reason: str
    escalation_summary: str
    response: str
    tools_used: List[str]
    next_agent: str
    conversation_history: List[Any]
    # V15（2026-08-14）：memory 字段已删——原 TypedDict 声明 + classify 初始化 None 后
    # 全项目零写入（base_agent 用 session_manager 兜底，无 LangChain memory 注入路径），死字段
    # 由图 checkpointer 持久化，跨 LangGraph 工作进程仍可续聊（内存 session_manager 无法做到）
    persisted_dialogue: List[Any]
    # V12 滚动摘要：persisted_dialogue 超 DIALOGUE_CAP 后，被归档轮次浓缩为一句摘要，
    # prompt 注入 = 摘要 + 尾部近 12 条（旧轮完整原文存归档表，web 渲染合并不丢历史）
    dialogue_summary: str
    # A5 复合查询（2026-08-17）：detect_compound 命中且拆解出 ≥2 子查询时写入，
    # 由 route_after_classify 优先路由到 compound 节点（综合回答，不按单标签升级）
    compound_subs: List[Any]
    # 通道故障降级标记（韧性路由依据，2026-09-15 故障注入 FI-11 补声明）：classify 检测
    # LLMServiceUnavailable 时置 True，route_after_classify 据此直连 final_response
    # （不进业务 agent 二次撞故障）。必须声明为状态通道——不声明则 LangGraph 合并时
    # 丢弃该键，降级路由只能靠未声明键透传的实现细节侥幸工作，且结果不可观测。
    llm_unavailable: bool
