"""业务智能体节点工厂 + 初始化。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

import logging
from datetime import datetime

from llm import get_llm
from citations import _attach_citations
from multi_agents import (
    ProductAgent, TechAgent, BillingAgent,
    ComplaintAgent, GeneralAgent
)
from session_manager import default_session_manager  # noqa: F401（供 agent.set_session_manager 兜底注入）
from .dialogue import _compact_after_append

logger = logging.getLogger(__name__)


# 初始化智能体
def initialize_agents():
    """初始化所有智能体"""
    agents = {
        "product_agent": ProductAgent(),
        "tech_agent": TechAgent(),
        "billing_agent": BillingAgent(),
        "complaint_agent": ComplaintAgent(),
        "general_agent": GeneralAgent()
    }

    # 为每个智能体设置LLM和会话管理器
    for agent in agents.values():
        agent.set_llm(get_llm())  # 延迟获取LLM
        agent.set_session_manager(default_session_manager)

    return agents


# 定义智能体处理节点
def create_agent_node(agent_name: str):
    """创建智能体处理节点"""
    def agent_node(state: dict) -> dict:
        agents = initialize_agents()
        agent = agents.get(agent_name)
        if agent:
            # 获取会话上下文
            session_id = state["session_id"]
            state["conversation_history"] = list(state.get("persisted_dialogue") or [])

            # 处理查询
            result = agent.process(state)

            if not isinstance(result, dict):
                logger.error(f"Agent {agent_name} 返回非 dict: {type(result)}")
                result = {"response": "Error: Agent processing failed", "current_agent": agent_name}

            if "response" not in result:
                logger.error(f"Agent {agent_name} 结果缺 response 字段: {str(result)[:200]}")
                result["response"] = "Error: No response from agent"

            # A6 引用溯源（harness 统一附加，不碰 agent prompt）：正常应答且未升级时，
            # 按 agent 域重新取命中条目标题，在回答末尾标注"参考来源"（与 agent 内部检索
            # 同判定路径，见 kb_retriever.retrieve_titles）；引用脚注随回答一并持久化。
            if not result.get("escalated"):
                _attach_citations(agent_name, result)

            # 助手轮次写入 checkpointer 状态（V12：统一接入点触发压缩）
            _compact_after_append(result, {
                "content": str(result["response"]),
                "is_user": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            return result
        else:
            state["response"] = f"Error: Agent {agent_name} not found"
            return state
    return agent_node
