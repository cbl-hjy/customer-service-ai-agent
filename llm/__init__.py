"""LLM 通道包：客户端 + 韧性层。

公共入口统一从 `llm` 导入（get_llm / get_token_usage / reset_token_usage /
CircuitBreaker / OpenAICompatibleClient / CustomResponse）。
私有单例（_TOKEN_USAGE / _TOKEN_LOCK / _circuit_breaker / _API_SEMAPHORE /
_THINKING_ONLY_SNAPSHOTS）一并 re-export：trace 采集与既有测试依赖对象同一性
（re-export 的是同一对象，不是拷贝）。
"""

from .resilience import (  # noqa: F401
    _API_SEMAPHORE,
    _THINKING_ONLY_SNAPSHOTS,
    _TOKEN_LOCK,
    _TOKEN_USAGE,
    _circuit_breaker,
    CircuitBreaker,
    get_channel_status,
    get_token_usage,
    llm_slot,
    reset_token_usage,
    _record_token_usage,
)
from .client import (  # noqa: F401
    CustomResponse,
    OpenAICompatibleClient,
    _llm_instance,
    get_llm,
    initialize_llm_client,
)
from .stream_sink import (  # noqa: F401  P1 流式通道（SSE）：sink 注册 + call_llm 统一入口
    call_llm,
    clear_sink,
    get_sink,
    push,
    set_sink,
)
