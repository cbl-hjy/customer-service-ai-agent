"""流式通道（SSE）：thread-local sink 注册表。

2026-08-18 产品化（P1 真流式）设计：
- token 流：web 层在 worker 线程注册 sink → 图节点（同线程执行）内 LLM 逐 token 推送
- 阶段流：_traced 节点包装器在节点开始时推送 stage（复用既有旁路，零业务侵入）
- 隔离：thread-local——Flask 多请求并发下各 worker 线程 sink 互不串扰（与 V6 同纪律）
- opt-in：未注册 sink 时 call_llm 走原 invoke，行为零变化（评估/单测/CLI 不受影响）
"""

import logging
import threading

from exceptions import LLMServiceUnavailable

logger = logging.getLogger(__name__)

_local = threading.local()


def set_sink(fn) -> None:
    """注册当前线程的流式 sink：fn(kind, payload)，kind ∈ {"token", "stage"}。"""
    _local.sink = fn


def clear_sink() -> None:
    """清除当前线程 sink（worker 结束时必须调用，防线程池复用残留）。"""
    _local.sink = None


def get_sink():
    """读取当前线程 sink（未注册返回 None——图节点在 web worker 线程内执行，天然可见）。"""
    return getattr(_local, "sink", None)


def push(kind: str, payload) -> None:
    """非阻塞推送：sink 不存在（非流式路径）或推送异常时静默跳过（流式是增强不是依赖）。"""
    fn = getattr(_local, "sink", None)
    if fn is None:
        return
    try:
        fn(kind, payload)
    except Exception:  # noqa: BLE001  sink 故障不影响主流程
        pass


def call_llm(llm, messages):
    """正文生成的统一入口：注册了 sink → 流式逐 token 推送 + 聚合全文；
    未注册 → 原 invoke（行为零变化）。返回与 invoke 同构的响应对象（.content）。

    显式 opt-in 边界：只有 agent 正文生成与 A5 综合回答走本入口——
    分类（response_format=json）/A4 改写/A5 拆解的中间产物不推送给用户。

    零 token 兜底（可用性优先）：流式请求在产出任何 token 前失败（如服务端
    不接受流式参数/瞬时故障重试耗尽）→ 回退非流式 invoke。用户侧表现为
    无流式效果但拿到完整回复，优于直接报错。已产出 token 后失败不可兜底
    （重放会造成内容重复），异常正常上抛由 agent 层降级。
    """
    if get_sink() is None:
        return llm.invoke(messages)
    parts = []
    try:
        for token in llm.invoke_stream(messages):
            parts.append(token)
            push("token", token)
    except LLMServiceUnavailable:
        if parts:
            raise  # 已交付部分内容，不可重放
        logger.warning("流式调用零 token 失败，回退非流式 invoke")
        return llm.invoke(messages)
    return _AggregatedResponse("".join(parts))


def call_llm_with_tools(llm, messages, tools):
    """T1 工具感知的正文生成入口（2026-09-16）：返回 .content + .tool_calls。

    - sink 未注册（评估/单测/CLI）→ llm.invoke(messages, tools)（与 T1 首版行为一致）；
    - sink 注册（web 流式）→ invoke_stream_tools：纯文本轮打字机直推，
      工具轮前导抑制 + tool_calls 分片聚合（三层策略见 _ToolStream docstring）；
    - 客户端无 invoke_stream_tools（mock/旧实现）→ 回退 invoke（兼容）；
    - 零产出失败回退非流式（与 call_llm 同兜底纪律；已产出或已聚合则不可重放）。
    """
    if get_sink() is None:
        return llm.invoke(messages, tools=tools)
    stream_factory = getattr(llm, "invoke_stream_tools", None)
    if stream_factory is None:
        return llm.invoke(messages, tools=tools)
    ts = stream_factory(messages, tools)
    parts = []
    try:
        for token in ts:
            parts.append(token)
            push("token", token)
    except LLMServiceUnavailable:
        if parts or ts.tool_calls:
            raise  # 已交付内容/工具分片，不可重放
        logger.warning("工具流式调用零产出失败，回退非流式 invoke")
        return llm.invoke(messages, tools=tools)
    return _AggregatedResponse("".join(parts), tool_calls=ts.tool_calls)


class _AggregatedResponse:
    """流式聚合结果：与 CustomResponse 同构（.content + .tool_calls），调用方无感知。"""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
