"""多智能体客服系统（兼容壳）。

2026-08-18 拆包重构：原 1225 行单体拆为 llm/（通道+韧性）、nodes/（业务节点）、
graph/（图组装+checkpointer）、citations.py（引用溯源）、trace.py（追踪）四个模块，
本文件仅作 re-export 兼容层——既有 import（eval/web/bench/verify/tests）零改动。
行为零变化：代码逐字迁移，223 测试全绿验收（详见 ai-docs/w9 报告）。

图入口保持 "./multi_agent_customer_service.py:make_graph"（langgraph.json 不变）。
"""

import os

# fail-safe：限制 checkpoint 反序列化为安全类型（幂等；graph.checkpointer 同样设置）
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from dotenv import load_dotenv

load_dotenv()

# ---- LLM 通道（客户端 + 韧性：熔断/token 计量/并发门控） ----
from llm import (  # noqa: F401,F403
    CircuitBreaker,
    CustomResponse,
    OpenAICompatibleClient,
    _API_SEMAPHORE,
    _THINKING_ONLY_SNAPSHOTS,
    _TOKEN_LOCK,
    _TOKEN_USAGE,
    _circuit_breaker,
    _llm_instance,
    get_llm,
    get_token_usage,
    initialize_llm_client,
    reset_token_usage,
)

# ---- 通道异常（错误分类：系统故障 vs 分类不确定） ----
from exceptions import LLMServiceUnavailable  # noqa: F401

# ---- 状态与常量 ----
from config import (  # noqa: F401
    DIALOGUE_CAP,
    DIALOGUE_TAIL_KEEP,
    MAX_CONCURRENCY,
)
from nodes.state import AgentState, OUT_OF_SCOPE_REPLY  # noqa: F401

# ---- 节点（分类/复合/业务智能体/升级/最终响应 + V12 压缩） ----
from nodes import (  # noqa: F401
    _COMPOUND_SYSTEM_PROMPT,
    _compact_after_append,
    _compact_dialogue_if_needed,
    _compound_escalate,
    _compound_retrieve_all,
    _DIALOGUE_SUMMARY_MAX_CHARS,
    _try_decompose,
    classify_query_node,
    compound_node,
    create_agent_node,
    escalate_node,
    final_response_node,
    initialize_agents,
)

# ---- 引用溯源（A6/A5） ----
from citations import (  # noqa: F401
    _AGENT_DOMAIN_MAP,
    _CITATION_FOOTER_PREFIX,
    _attach_citations,
    _attach_compound_citations,
    ENABLE_CITATIONS,
)

# ---- V18 全链路追踪 ----
from trace import _traced  # noqa: F401

# ---- 图组装 + checkpointer ----
from graph import (  # noqa: F401
    CHECKPOINT_DIR,
    _make_query_decomposer,
    _make_query_rewriter,
    get_checkpointer,
    make_graph,
)

# ---- 分类工具与路由（旧 import 兼容） ----
from tools import classify_query  # noqa: F401
from routing import should_escalate, wants_human  # noqa: F401
from session_manager import default_session_manager  # noqa: F401
from multi_agents import (  # noqa: F401
    ProductAgent, TechAgent, BillingAgent,
    ComplaintAgent, GeneralAgent
)


# 创建默认工作流实例
if __name__ == "__main__":
    app = make_graph()
    print("多智能体客服系统启动成功")
