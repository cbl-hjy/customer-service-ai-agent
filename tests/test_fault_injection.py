"""测试：LLM 通道故障注入（质量收口，2026-09-15）。

注入纪律（混沌工程边界注入）：全部故障在 requests.post 传输缝隙注入——
确定性、密封、不烧 API 额度；断言双层面 = 用户可见行为 + 通道内部不变量
（网络调用次数/熔断状态/token 计量）。

具名风险清单：
  R1  429+Retry-After 契约：精确等待（ra ~ 1.3×ra jitter）后重试成功，熔断不记失败
  R2  429 无 Retry-After：指数退避（2^attempt ± 30% jitter）重试成功
  R3  429+Retry-After 重试耗尽：必须抛 LLMServiceUnavailable + 熔断记失败
      （非流式/流式两条路径；曾暴露缺陷：末次尝试 continue 跳过耗尽检查 → invoke 静默返回 None）
  R4  超时（ReadTimeout）：瞬时故障退避重试后成功
  R5  超时重试耗尽：抛 LLMServiceUnavailable（status=None）+ 熔断记失败
  R6  401 永久故障：单次调用不重试 + 熔断记失败 + status_code 透传
  R7  5xx 瞬时故障：退避重试后成功，熔断不记失败
  R8  熔断 OPEN：快速失败，零网络调用
  R9  连续失败把熔断器推到 OPEN：后续 invoke 零网络调用（额度保护）
  R10 失败路径不计 token（成本基线不被故障污染）
  R11 图级降级链：分类通道 403 → 全图跑通 → 友好降级文案（不异常/不升级/llm_unavailable 路由）
  R12 SSE 端点级：LLM 全故障 → done 帧带降级文案（非 error 帧）
  R13 熔断恢复链：OPEN 冷却 → HALF_OPEN 探测成功 → CLOSED（经 client 真实路径）
"""

import json
import sys
import os
import time

import pytest
import requests as _requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exceptions import LLMServiceUnavailable
from llm.client import OpenAICompatibleClient


# ---------------------------------------------------------------------------
# 隔离与桩设施
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_channel_state():
    """每测前后复位全局通道状态（熔断器/token 计数），与 test_stream_sse 同纪律。"""
    from llm.resilience import _circuit_breaker, reset_token_usage

    def _reset():
        _circuit_breaker._state = "CLOSED"
        _circuit_breaker._failure_count = 0
        _circuit_breaker._success_count = 0
        reset_token_usage()

    _reset()
    yield
    _reset()


class _FakeResponse:
    """模拟 requests 非流式响应：可配状态码/headers/JSON 体。"""

    def __init__(self, status_code=200, content="ok", text="", headers=None):
        self.status_code = status_code
        self.text = text or ("" if status_code < 400 else f"http {status_code}")
        self.headers = headers
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _requests.exceptions.HTTPError(
                f"{self.status_code} Error", response=self
            )

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }


def _make_client(retries=3):
    c = OpenAICompatibleClient(api_key="test-key", base_url="http://fake", model="test-model")
    c.max_retries = retries
    return c


def _install_transport(monkeypatch, responses, sleeps):
    """注入传输故障序列 + 打桩 sleep（记录不真等）。

    responses: 每次 requests.post 的返回/异常（ callable -> response 或 raise）；
    越界时重复最后一个元素（恒故障场景）。
    """
    import llm.client as _lc

    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if callable(item):
            return item()
        return item

    monkeypatch.setattr(_lc.requests, "post", fake_post)
    monkeypatch.setattr(_lc.time, "sleep", lambda s: sleeps.append(s))
    return calls


# ---------------------------------------------------------------------------
# R1-R10：invoke() 非流式通道故障矩阵
# ---------------------------------------------------------------------------

def test_r1_429_retry_after_contract(monkeypatch):
    """R1：429 带 Retry-After → 精确等待（ra~1.3ra）后重试成功，熔断不记失败。"""
    from llm.resilience import _circuit_breaker

    sleeps = []
    responses = [
        _FakeResponse(status_code=429, headers={"Retry-After": "2"}),
        _FakeResponse(status_code=200, content="恢复了"),
    ]
    calls = _install_transport(monkeypatch, responses, sleeps)

    resp = _make_client().invoke([{"role": "user", "content": "q"}])

    assert resp.content == "恢复了"
    assert len(calls) == 2, "429 后应重试一次"
    assert len(sleeps) == 1 and 2.0 <= sleeps[0] <= 2.6, f"Retry-After 等待应 ∈[2, 2.6]，实际 {sleeps}"
    assert _circuit_breaker._failure_count == 0, "重试后成功的 429 不应记熔断失败"


def test_r2_429_without_retry_after_backoff(monkeypatch):
    """R2：429 无 Retry-After → 指数退避（首退避 2^0±30% = 1~1.3s）重试成功。"""
    sleeps = []
    responses = [
        _FakeResponse(status_code=429, headers={}),  # 有 headers 但无 Retry-After
        _FakeResponse(status_code=200, content="ok"),
    ]
    calls = _install_transport(monkeypatch, responses, sleeps)

    resp = _make_client().invoke([{"role": "user", "content": "q"}])

    assert resp.content == "ok"
    assert len(calls) == 2
    assert len(sleeps) == 1 and 1.0 <= sleeps[0] <= 1.3, f"退避应 ∈[1, 1.3]，实际 {sleeps}"


def test_r3_429_retry_after_exhausted_raises(monkeypatch):
    """R3（缺陷复现→已修复）：429+Retry-After 重试耗尽必须抛 LLMServiceUnavailable。

    曾暴露缺陷：末次尝试走 429 continue 分支跳过耗尽检查 → for 循环裸结束 →
    invoke 静默返回 None，上层把它伪装成"分类解析失败"降级。
    """
    from llm.resilience import _circuit_breaker

    sleeps = []
    responses = [_FakeResponse(status_code=429, headers={"Retry-After": "1"})]
    calls = _install_transport(monkeypatch, responses, sleeps)

    with pytest.raises(LLMServiceUnavailable) as ei:
        _make_client(retries=3).invoke([{"role": "user", "content": "q"}])

    assert ei.value.status_code == 429
    assert len(calls) == 3, "重试耗尽 = max_retries 次网络调用"
    assert _circuit_breaker._failure_count >= 1, "耗尽必须熔断记失败"


def test_r3s_stream_429_retry_after_exhausted_raises(monkeypatch):
    """R3 流式路径：429+Retry-After 重试耗尽必须抛，不得静默产出空流。"""
    from llm.resilience import _circuit_breaker

    sleeps = []
    lines = [b"data: [DONE]", b""]
    responses = [
        _FakeResponse(status_code=429, headers={"Retry-After": "1"}),
    ]

    # 流式响应需要 iter_lines
    class _Stream429(_FakeResponse):
        def iter_lines(self):
            yield from lines

    responses = [_Stream429(status_code=429, headers={"Retry-After": "1"})]
    calls = _install_transport(monkeypatch, responses, sleeps)

    client = _make_client(retries=3)
    with pytest.raises(LLMServiceUnavailable):
        list(client.invoke_stream([{"role": "user", "content": "q"}]))

    assert len(calls) == 3
    assert _circuit_breaker._failure_count >= 1


def test_r4_timeout_retries_then_succeeds(monkeypatch):
    """R4：读超时（瞬时）→ 退避重试后成功。"""
    sleeps = []
    responses = [
        lambda: (_ for _ in ()).throw(_requests.exceptions.ReadTimeout("read timed out")),
        _FakeResponse(status_code=200, content="慢但成功"),
    ]
    calls = _install_transport(monkeypatch, responses, sleeps)

    resp = _make_client().invoke([{"role": "user", "content": "q"}])

    assert resp.content == "慢但成功"
    assert len(calls) == 2
    assert len(sleeps) == 1, "瞬时故障退避一次"


def test_r5_timeout_exhausted_raises_and_records(monkeypatch):
    """R5：超时重试耗尽 → 抛 LLMServiceUnavailable(status=None) + 熔断记失败。"""
    from llm.resilience import _circuit_breaker

    sleeps = []
    responses = [
        lambda: (_ for _ in ()).throw(_requests.exceptions.ReadTimeout("read timed out")),
    ]
    calls = _install_transport(monkeypatch, responses, sleeps)

    with pytest.raises(LLMServiceUnavailable) as ei:
        _make_client(retries=3).invoke([{"role": "user", "content": "q"}])

    assert ei.value.status_code is None, "超时无 HTTP 状态码"
    assert len(calls) == 3
    assert _circuit_breaker._failure_count >= 1


def test_r6_401_permanent_no_retry(monkeypatch):
    """R6：401 永久故障 → 单次调用不重试 + 熔断记失败 + status_code 透传。"""
    from llm.resilience import _circuit_breaker

    sleeps = []
    responses = [_FakeResponse(status_code=401, text="unauthorized")]
    calls = _install_transport(monkeypatch, responses, sleeps)

    with pytest.raises(LLMServiceUnavailable) as ei:
        _make_client().invoke([{"role": "user", "content": "q"}])

    assert ei.value.status_code == 401
    assert len(calls) == 1, "永久故障不重试"
    assert not sleeps, "永久故障不等待"
    assert _circuit_breaker._failure_count == 1


def test_r7_5xx_retries_then_succeeds(monkeypatch):
    """R7：5xx 瞬时故障 → 退避重试后成功，熔断不记失败。"""
    from llm.resilience import _circuit_breaker

    sleeps = []
    responses = [
        _FakeResponse(status_code=502, text="bad gateway"),
        _FakeResponse(status_code=200, content="恢复了"),
    ]
    calls = _install_transport(monkeypatch, responses, sleeps)

    resp = _make_client().invoke([{"role": "user", "content": "q"}])

    assert resp.content == "恢复了"
    assert len(calls) == 2
    assert _circuit_breaker._failure_count == 0, "重试后成功不记失败"


def test_r8_circuit_open_fast_fail_zero_calls(monkeypatch):
    """R8：熔断 OPEN → 快速失败，零网络调用（额度保护）。"""
    import llm.client as _lc
    from llm.resilience import _circuit_breaker

    _circuit_breaker._state = "OPEN"
    calls = []
    monkeypatch.setattr(_lc.requests, "post", lambda *a, **k: calls.append(1))

    with pytest.raises(LLMServiceUnavailable) as ei:
        _make_client().invoke([{"role": "user", "content": "q"}])

    assert "熔断" in str(ei.value)
    assert not calls, "熔断 OPEN 不得发网络请求"


def test_r9_repeated_exhaustion_trips_breaker(monkeypatch):
    """R9：连续失败把熔断器推到 OPEN（经真实 invoke 路径），后续快速失败。"""
    from llm.resilience import _circuit_breaker

    sleeps = []
    responses = [
        lambda: (_ for _ in ()).throw(_requests.exceptions.ConnectionError("down")),
    ]
    calls = _install_transport(monkeypatch, responses, sleeps)

    client = _make_client(retries=2)
    # 阈值 3（config 默认）：每轮 invoke 耗尽 2 次重试记 1 次失败
    for _ in range(3):
        with pytest.raises(LLMServiceUnavailable):
            client.invoke([{"role": "user", "content": "q"}])
    assert _circuit_breaker._state == "OPEN", "3 次连续失败应熔断"

    n_calls_before = len(calls)
    with pytest.raises(LLMServiceUnavailable):
        client.invoke([{"role": "user", "content": "q"}])
    assert len(calls) == n_calls_before, "熔断后不再消耗网络调用/API 额度"


def test_r10_failures_do_not_record_token_usage(monkeypatch):
    """R10：失败路径不计 token（成本基线不被故障污染）。"""
    from llm.resilience import get_token_usage

    sleeps = []
    responses = [_FakeResponse(status_code=403, text="quota")]
    _install_transport(monkeypatch, responses, sleeps)

    with pytest.raises(LLMServiceUnavailable):
        _make_client().invoke([{"role": "user", "content": "q"}])

    usage = get_token_usage()
    assert usage["calls"] == 0 and usage["total_tokens"] == 0


# ---------------------------------------------------------------------------
# R11-R12：图级 / SSE 端点级降级链（注入传输 403，跑真实图）
# ---------------------------------------------------------------------------

@pytest.fixture()
def _faulted_graph(monkeypatch, tmp_path):
    """LLM 通道全故障（403）的真实图：InMemory checkpointer + 真实降级链路。

    make_graph 会全局注入 A4 改写器/A5 拆解器（配对 clear 纪律，用完卸载）。
    """
    import llm.client as _lc
    import graph.builder as _gb
    from langgraph.checkpoint.memory import InMemorySaver
    from kb_retriever import clear_query_rewriter
    from tools.query_decompose import clear_query_decomposer

    monkeypatch.setattr(_gb, "get_checkpointer", lambda: InMemorySaver())
    monkeypatch.setattr(
        _lc.requests, "post",
        lambda *a, **k: _FakeResponse(status_code=403, text="quota exhausted"),
    )
    app = _gb.make_graph()
    try:
        yield app
    finally:
        clear_query_rewriter()
        clear_query_decomposer()


def test_r11_graph_degrades_gracefully_on_403(_faulted_graph):
    """R11：分类通道 403 → 全图跑通 → 友好降级文案，不异常/不升级/不走业务 agent。"""
    result = _faulted_graph.invoke(
        {"customer_query": "你们最新款手机多少钱？"},
        {"configurable": {"thread_id": f"fi-r11-{time.time()}"}},
    )

    assert result.get("llm_unavailable") is True, "通道故障必须打标记（路由依据）"
    assert "暂时不可用" in result.get("response", ""), "应为友好降级文案"
    assert result.get("escalated") is False, "通道故障不升级（升级了人工也调不了 LLM）"
    assert result.get("current_agent") == "智能客服"
    assert result.get("query_type") == "general_inquiry"


def test_r12_sse_delivers_degradation_done_frame(_faulted_graph, monkeypatch):
    """R12：SSE 端点 LLM 全故障 → done 帧带降级文案（非 error 帧），流完整收口。"""
    import chat_web_service as cws

    monkeypatch.setattr(cws, "get_app", lambda: _faulted_graph)
    chunks = list(cws.run_chat_stream_events("你们最新款手机多少钱？", f"fi-r12-{time.time()}"))

    events = []
    for ch in chunks:
        for line in ch.split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    assert "error" not in types, "通道故障是可降级场景，不应走 error 帧"
    done = next((e for e in events if e["type"] == "done"), None)
    assert done is not None and "暂时不可用" in done["content"], "done 帧应带降级文案"
    assert "".join(chunks).endswith("data: [DONE]\n\n"), "流必须完整收口"


def test_r13_breaker_recovery_through_client(monkeypatch):
    """R13：OPEN 冷却到期 → client 放行探测（HALF_OPEN）→ 成功 → CLOSED。"""
    from llm.resilience import _circuit_breaker

    sleeps = []
    responses = [_FakeResponse(status_code=200, content="恢复")]
    _install_transport(monkeypatch, responses, sleeps)

    breaker = _circuit_breaker
    breaker._state = "OPEN"
    breaker._last_failure_time = time.time() - 9999  # 冷却早已到期

    resp = _make_client().invoke([{"role": "user", "content": "q"}])

    assert resp.content == "恢复"
    assert breaker._state == "CLOSED", "探测成功应恢复 CLOSED"
