"""最终响应节点。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""


def final_response_node(state: dict) -> dict:
    """Generate final response（直接透传，不向客户暴露内部 agent 前缀）。

    原实现会拼 【{agent}'s Response】 前缀，客户可见即缺陷（内部泄漏），
    且让 web 层 response 与 persisted_dialogue 正文不一致，导致去重失败、双线气泡。
    现直接返回业务 agent / escalate / 护栏生成的正文。
    """
    return state
