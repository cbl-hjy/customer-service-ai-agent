"""测试：P1 真流式通道（SSE，2026-08-21）
- invoke_stream：SSE 行解析 / usage 计量 / 4xx 永久故障 / 首 token 前后重试语义
- stream_sink.call_llm：opt-in 边界（无 sink 走 invoke）、token 推送聚合、零 token 兜底
- thread-local sink 隔离（并发请求不串扰，V6 同纪律）
- _traced 阶段流推送
- run_chat_stream_events：SSE 帧协议端到端（meta → stage/token → done → [DONE]）
"""

import json
import sys
import os
import threading

import pytest
import requests as _requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exceptions import LLMServiceUnavailable
from llm import stream_sink
from llm.client import OpenAICompatibleClient


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_channel_state():
    """每测前后都复位全局通道状态（熔断器/token 计数）。

    前置复位必须：同进程先跑的其他测试文件（如 v14 真实图调用）会累积
    _TOKEN_USAGE，不前置复位则绝对值断言被跨文件污染。
    """
    from llm.resilience import _circuit_breaker, reset_token_usage

    def _reset():
        _circuit_breaker._state = "CLOSED"
        _circuit_breaker._failure_count = 0
        reset_token_usage()

    _reset()
    yield
    _reset()


class _FakeStreamResponse:
    """模拟 requests 流式响应：iter_lines 逐行吐 SSE 字节行。"""

    def __init__(self, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _requests.exceptions.HTTPError(
                f"{self.status_code} Error", response=self
            )

    def iter_lines(self):
        for line in self._lines:
            yield line


def _sse_lines(tokens=("你", "好"), with_usage=True):
    """构造标准 SSE 字节行序列。"""
    lines = []
    for t in tokens:
        lines.append(b'data: ' + json.dumps({"choices": [{"delta": {"content": t}}]}).encode())
        lines.append(b"")
    if with_usage:
        usage = {"prompt_tokens": 10, "completion_tokens": len(tokens)}
        lines.append(b'data: ' + json.dumps({"choices": [{"delta": {}}], "usage": usage}).encode())
        lines.append(b"")
    lines.append(b"data: [DONE]")
    lines.append(b"")
    return lines


def _make_client():
    return OpenAICompatibleClient(api_key="test-key", base_url="http://fake", model="test-model")


# ---------------------------------------------------------------------------
# invoke_stream：SSE 解析 + 韧性契约
# ---------------------------------------------------------------------------

def test_invoke_stream_yields_tokens_and_records_usage(monkeypatch):
    """正常流：逐 token yield、usage 计量（stream_options 末 chunk）、熔断记成功。"""
    import llm.client as _lc
    from llm.resilience import _circuit_breaker, get_token_usage

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, stream=None):
        captured["payload"] = json
        return _FakeStreamResponse(lines=_sse_lines(("你", "好", "呀")))

    monkeypatch.setattr(_lc.requests, "post", fake_post)
    client = _make_client()

    tokens = list(client.invoke_stream([{"role": "user", "content": "hi"}]))

    assert tokens == ["你", "好", "呀"]
    # 流式 payload 契约
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["stream_options"] == {"include_usage": True}
    # usage 计量
    usage = get_token_usage()
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 3
    # 熔断记成功
    assert _circuit_breaker._state == "CLOSED"


def test_invoke_stream_skips_bad_lines_and_done(monkeypatch):
    """坏 JSON 行 / [DONE] / 空 delta 不终止流也不产出。"""
    import llm.client as _lc

    lines = [
        b": ping",  # 注释行
        b"data: not-json",
        b"",
        b'data: {"choices":[{"delta":{"content":"A"}}]}',
        b"",
        b'data: {"choices":[{"delta":{}}]}',  # 空 delta
        b"",
        b'data: {"usage":{"prompt_tokens":5,"completion_tokens":1}}',  # 纯 usage chunk 无 choices
        b"",
        b"data: [DONE]",
        b"",
    ]
    monkeypatch.setattr(_lc.requests, "post", lambda *a, **k: _FakeStreamResponse(lines=lines))
    client = _make_client()

    tokens = list(client.invoke_stream([{"role": "user", "content": "hi"}]))
    assert tokens == ["A"]


def test_invoke_stream_4xx_permanent_no_retry(monkeypatch):
    """4xx 永久故障（403）：不重试（只发一次请求），熔断记失败。"""
    import llm.client as _lc
    from llm.resilience import _circuit_breaker

    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return _FakeStreamResponse(status_code=403, text="quota exhausted")

    monkeypatch.setattr(_lc.requests, "post", fake_post)
    client = _make_client()

    with pytest.raises(LLMServiceUnavailable) as ei:
        list(client.invoke_stream([{"role": "user", "content": "hi"}]))

    assert ei.value.status_code == 403
    assert len(calls) == 1, "4xx 永久故障不应重试"
    assert _circuit_breaker._failure_count >= 1


def test_invoke_stream_retries_before_first_token(monkeypatch):
    """首 token 前瞬时故障（连接错误）→ 重试后成功（退避 sleep 打桩防慢测）。"""
    import llm.client as _lc

    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise _requests.exceptions.ConnectionError("boom")
        return _FakeStreamResponse(lines=_sse_lines(("好",)))

    monkeypatch.setattr(_lc.requests, "post", fake_post)
    monkeypatch.setattr(_lc.time, "sleep", lambda s: None)
    client = _make_client()
    client.max_retries = 3

    tokens = list(client.invoke_stream([{"role": "user", "content": "hi"}]))
    assert tokens == ["好"]
    assert len(calls) == 2, "瞬时故障应在首 token 前重试"


def test_invoke_stream_no_retry_after_first_token(monkeypatch):
    """首 token 之后断流：不可重试（防内容重复），直接抛 LLMServiceUnavailable。"""
    import llm.client as _lc

    calls = []

    class _BreakMidStream(_FakeStreamResponse):
        def iter_lines(self):
            yield b'data: ' + json.dumps({"choices": [{"delta": {"content": "部"}}]}).encode()
            yield b""
            raise _requests.exceptions.ChunkedEncodingError("connection dropped")

    def fake_post(*a, **k):
        calls.append(1)
        return _BreakMidStream()

    monkeypatch.setattr(_lc.requests, "post", fake_post)
    monkeypatch.setattr(_lc.time, "sleep", lambda s: None)
    client = _make_client()

    gen = client.invoke_stream([{"role": "user", "content": "hi"}])
    assert next(gen) == "部"  # 首 token 已交付
    with pytest.raises(LLMServiceUnavailable):
        next(gen)
    assert len(calls) == 1, "已交付 token 后不应重试"


def test_invoke_stream_blocked_when_circuit_open():
    """熔断 OPEN：快速失败，不发网络请求。"""
    from llm.resilience import _circuit_breaker
    _circuit_breaker._state = "OPEN"
    client = _make_client()

    with pytest.raises(LLMServiceUnavailable):
        list(client.invoke_stream([{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# stream_sink：opt-in 边界 + token 推送 + 兜底
# ---------------------------------------------------------------------------

class _StubLLM:
    """双模式桩：invoke 返回全文；invoke_stream 逐 token（可在产出若干 token 后抛错）。"""

    def __init__(self, full="全文", tokens=("全", "文"), stream_error=None, error_after=0):
        self.full = full
        self.tokens = tokens
        self.stream_error = stream_error
        self.error_after = error_after  # 产出 N 个 token 后抛 stream_error（0=开头即抛）
        self.invoke_called = 0
        self.stream_called = 0

    def invoke(self, messages, response_format=None):
        self.invoke_called += 1
        return type("R", (), {"content": self.full})()

    def invoke_stream(self, messages, response_format=None):
        self.stream_called += 1
        for i, t in enumerate(self.tokens):
            if self.stream_error is not None and i >= self.error_after:
                raise self.stream_error
            yield t
        if self.stream_error is not None and self.error_after >= len(self.tokens):
            raise self.stream_error


def test_call_llm_without_sink_uses_invoke():
    """opt-in：未注册 sink 走原 invoke（评估/单测/CLI 行为零变化）。"""
    stream_sink.clear_sink()
    llm = _StubLLM()
    resp = stream_sink.call_llm(llm, [{"role": "user", "content": "q"}])
    assert resp.content == "全文"
    assert llm.invoke_called == 1
    assert llm.stream_called == 0


def test_call_llm_with_sink_streams_and_aggregates():
    """注册 sink：逐 token 推送 + 聚合全文（.content 与 invoke 同构）。"""
    events = []
    stream_sink.set_sink(lambda kind, payload: events.append((kind, payload)))
    try:
        llm = _StubLLM(tokens=("您", "好", "！"))
        resp = stream_sink.call_llm(llm, [{"role": "user", "content": "q"}])
        assert resp.content == "您好！"
        assert events == [("token", "您"), ("token", "好"), ("token", "！")]
        assert llm.invoke_called == 0
    finally:
        stream_sink.clear_sink()


def test_call_llm_zero_token_failure_falls_back_to_invoke():
    """零 token 流式失败 → 非流式 invoke 兜底（可用性优先）。"""
    stream_sink.set_sink(lambda kind, payload: None)
    try:
        llm = _StubLLM(stream_error=LLMServiceUnavailable("stream unsupported", status_code=400))
        resp = stream_sink.call_llm(llm, [{"role": "user", "content": "q"}])
        assert resp.content == "全文"
        assert llm.invoke_called == 1
    finally:
        stream_sink.clear_sink()


def test_call_llm_failure_after_tokens_reraises():
    """已推送 token 后失败：不可兜底（重放会重复），异常上抛。"""
    stream_sink.set_sink(lambda kind, payload: None)
    try:
        llm = _StubLLM(
            tokens=("您", "好"),
            stream_error=LLMServiceUnavailable("mid-stream drop", status_code=None),
            error_after=1,  # 产出 1 个 token 后断流
        )
        with pytest.raises(LLMServiceUnavailable):
            stream_sink.call_llm(llm, [{"role": "user", "content": "q"}])
        assert llm.invoke_called == 0
    finally:
        stream_sink.clear_sink()


def test_sink_thread_local_isolation():
    """thread-local 隔离：两线程各自注册的 sink 互不串扰（V6 纪律）。"""
    main_events = []
    other_events = []

    stream_sink.set_sink(lambda kind, payload: main_events.append((kind, payload)))
    try:
        def other_thread():
            stream_sink.set_sink(lambda kind, payload: other_events.append((kind, payload)))
            stream_sink.push("token", "other")
            stream_sink.clear_sink()

        t = threading.Thread(target=other_thread)
        t.start()
        t.join()

        stream_sink.push("token", "main")
    finally:
        stream_sink.clear_sink()

    assert main_events == [("token", "main")]
    assert other_events == [("token", "other")]


def test_push_without_sink_is_noop():
    """无 sink 时 push 静默跳过（非流式路径零开销）。"""
    stream_sink.clear_sink()
    stream_sink.push("token", "x")  # 不应抛错
    stream_sink.push("stage", {"node": "x"})


# ---------------------------------------------------------------------------
# _traced 阶段流
# ---------------------------------------------------------------------------

def test_traced_pushes_stage_on_node_start():
    """_traced 包装器在节点开始时推送 stage（sink 可见），节点行为不变。"""
    from trace import _traced, _trace_local

    events = []

    def node(state):
        return {"response": "ok"}

    wrapped = _traced("classify_query", node)
    stream_sink.set_sink(lambda kind, payload: events.append((kind, payload)))
    try:
        out = wrapped({"customer_query": "q", "session_id": "s-traced"})
        assert out["response"] == "ok"
    finally:
        stream_sink.clear_sink()
        # 配对清理：classify_query 入口节点会设置 thread-local run_id 且无出口节点
        # 清理——残留会污染同线程后续测试（v18_trace 的 start_run 幂等跳过，
        # fast 批缺 v14 真实图"清洗"时暴露，2026-09-16 定位）
        _trace_local.run_id = None

    stages = [p for k, p in events if k == "stage"]
    assert stages and stages[0]["node"] == "classify_query"


# ---------------------------------------------------------------------------
# run_chat_stream_events：SSE 帧协议端到端
# ---------------------------------------------------------------------------

def _parse_sse_frames(raw_chunks):
    """把生成器吐出的字符串拼起来，解析为事件列表（跳过心跳/结束标记）。"""
    text = "".join(raw_chunks)
    events = []
    for frame in text.split("\n\n"):
        data_line = None
        for line in frame.split("\n"):
            if line.startswith("data:"):
                data_line = line[len("data:"):].strip()
                break
        if data_line is None or data_line == "[DONE]":
            continue
        events.append(json.loads(data_line))
    return events, text


def test_run_chat_stream_events_frame_protocol(monkeypatch):
    """帧协议：meta 首帧 → stage/token（worker 内 sink 推送）→ done（权威全文）→ [DONE]。"""
    import chat_web_service as cws

    class _FakeApp:
        def invoke(self, graph_input, config):
            # 模拟图节点在 worker 线程内的行为：推 stage + 逐 token（走真实 sink 路径）
            stream_sink.push("stage", {"node": "classify_query"})
            stream_sink.push("stage", {"node": "general_agent"})
            for t in ("您", "好"):
                stream_sink.push("token", t)
            return {
                "response": "您好。（含引用脚注的权威全文）",
                "current_agent": "综合客服",
                "query_type": "general_inquiry",
                "escalated": False,
            }

    monkeypatch.setattr(cws, "get_app", lambda: _FakeApp())

    chunks = list(cws.run_chat_stream_events("你好", "sse-proto-1"))
    events, text = _parse_sse_frames(chunks)

    assert text.endswith("data: [DONE]\n\n"), "流必须以 [DONE] 结束"

    # 首帧 meta（会话键）
    assert events[0]["type"] == "meta"
    assert events[0]["session_id"] == "sse-proto-1"

    # stage 帧（顺序）
    stages = [e["node"] for e in events if e["type"] == "stage"]
    assert stages == ["classify_query", "general_agent"]

    # token 帧（顺序）
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert tokens == ["您", "好"]

    # done 帧（权威全文 + 决策字段）
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["content"] == "您好。（含引用脚注的权威全文）"
    assert done[0]["agent"] == "综合客服"
    assert done[0]["query_type"] == "general_inquiry"
    assert done[0]["escalated"] is False


def test_run_chat_stream_events_empty_message():
    """空消息：error 帧 + [DONE]，不启动 worker。"""
    import chat_web_service as cws

    chunks = list(cws.run_chat_stream_events("   ", "sse-empty"))
    events, text = _parse_sse_frames(chunks)

    assert events[0]["type"] == "error"
    assert "不能为空" in events[0]["error"]
    assert text.endswith("data: [DONE]\n\n")


def test_run_chat_stream_events_worker_error(monkeypatch):
    """worker 内图异常 → error 帧（不静默、不挂死）。"""
    import chat_web_service as cws

    class _BoomApp:
        def invoke(self, graph_input, config):
            raise RuntimeError("graph exploded")

    monkeypatch.setattr(cws, "get_app", lambda: _BoomApp())

    chunks = list(cws.run_chat_stream_events("你好", "sse-boom"))
    events, text = _parse_sse_frames(chunks)

    assert events[0]["type"] == "meta"
    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1
    assert "graph exploded" in errors[0]["error"]
    assert text.endswith("data: [DONE]\n\n")
