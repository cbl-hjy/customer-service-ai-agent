"""节点包：分类 / 复合 / 业务智能体 / 升级 / 最终响应。

统一从 `nodes` 导入节点函数（graph/builder.py 与兼容壳使用）。
"""

from .state import AgentState, OUT_OF_SCOPE_REPLY  # noqa: F401
from .dialogue import (  # noqa: F401
    _compact_dialogue_if_needed,
    _compact_after_append,
    _DIALOGUE_SUMMARY_MAX_CHARS,
)
from .classify import classify_query_node  # noqa: F401
from .compound import (  # noqa: F401
    compound_node,
    _try_decompose,
    _compound_retrieve_all,
    _compound_escalate,
    _COMPOUND_SYSTEM_PROMPT,
)
from .agents import initialize_agents, create_agent_node  # noqa: F401
from .escalate import escalate_node  # noqa: F401
from .final_response import final_response_node  # noqa: F401
