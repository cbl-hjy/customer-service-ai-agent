"""
测试：LLM 通道韧性（2026-08-13）
- 熔断器三状态（CLOSED / OPEN / HALF_OPEN）转换
- classify_query 异常分流：通道故障（LLMServiceUnavailable）不降级
- OpenAICompatibleClient 错误分类：403 永久故障不重试、5xx 重试
"""

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exceptions import LLMServiceUnavailable
from multi_agent_customer_service import CircuitBreaker


# ---------------------------------------------------------------------------
# 熔断器三状态
# ---------------------------------------------------------------------------

def test_circuit_breaker_closed_initial():
    cb = CircuitBreaker(fail_threshold=3, timeout=60, recovery_requests=1)
    assert cb._state == "CLOSED"
    assert cb.allow_request() is True


def test_circuit_breaker_trips_on_threshold():
    cb = CircuitBreaker(fail_threshold=3, timeout=60, recovery_requests=1)
    # 前 2 次失败未熔断
    cb.record_failure(500)
    assert cb._state == "CLOSED"
    cb.record_failure(500)
    assert cb._state == "CLOSED"
    # 第 3 次失败 → OPEN
    cb.record_failure(500)
    assert cb._state == "OPEN"
    # 熔断后不允许请求
    assert cb.allow_request() is False


def test_circuit_breaker_halves_after_timeout():
    cb = CircuitBreaker(fail_threshold=2, timeout=0.01, recovery_requests=1)
    cb.record_failure(500)
    cb.record_failure(500)
    assert cb._state == "OPEN"
    # 等冷却
    time.sleep(0.02)
    # 冷却后放行一个探测请求 → HALF_OPEN
    assert cb.allow_request() is True
    assert cb._state == "HALF_OPEN"


def test_circuit_breaker_recovers_on_success():
    cb = CircuitBreaker(fail_threshold=2, timeout=0.01, recovery_requests=1)
    cb.record_failure(500)
    cb.record_failure(500)
    assert cb._state == "OPEN"
    time.sleep(0.02)
    cb.allow_request()  # → HALF_OPEN
    cb.record_success()  # 探测成功 → CLOSED
    assert cb._state == "CLOSED"
    assert cb.allow_request() is True


def test_circuit_breaker_reopens_on_probe_failure():
    cb = CircuitBreaker(fail_threshold=2, timeout=0.01, recovery_requests=1)
    cb.record_failure(500)
    cb.record_failure(500)
    assert cb._state == "OPEN"
    time.sleep(0.02)
    cb.allow_request()  # → HALF_OPEN
    cb.record_failure(500)  # 探测失败 → 回 OPEN
    assert cb._state == "OPEN"


def test_circuit_breaker_success_resets_failures():
    cb = CircuitBreaker(fail_threshold=3, timeout=60, recovery_requests=1)
    cb.record_failure(500)
    cb.record_failure(500)
    cb.record_success()  # CLOSED 成功 → 清计数
    assert cb._failure_count == 0
    # 不再连续计数，需重新累计
    cb.record_failure(500)
    assert cb._state == "CLOSED"


# ---------------------------------------------------------------------------
# classify_query 异常分流：通道故障不降级
# ---------------------------------------------------------------------------

def test_classify_reraises_service_unavailable(monkeypatch):
    """LLMServiceUnavailable 必须原样抛出，不降级成 confidence=0"""
    from tools.query_tools import classify_query

    class _FailingLLM:
        def invoke(self, messages, response_format=None):
            raise LLMServiceUnavailable("LLM 通道返回 403", status_code=403)

    try:
        classify_query.invoke({"query": "手机多少钱", "llm": _FailingLLM()})
        assert False, "应当抛出 LLMServiceUnavailable"
    except LLMServiceUnavailable as e:
        assert e.status_code == 403


def test_classify_parse_error_falls_back():
    """非通道异常（如 llm 返回串）→ 降级 confidence=0（触发保守升级兜底）"""
    from tools.query_tools import classify_query

    class _BrokenLLM:
        def invoke(self, messages, response_format=None):
            from multi_agent_customer_service import CustomResponse
            return CustomResponse("not json at all")

    result = classify_query.invoke({"query": "手机多少钱", "llm": _BrokenLLM()})
    import json
    data = json.loads(result)
    assert data["confidence"] == 0.0
    assert data["label"] == "general_inquiry"


def test_llm_service_unavailable_has_status():
    e = LLMServiceUnavailable("额度耗尽", status_code=403)
    assert e.status_code == 403
    assert "额度耗尽" in str(e)


# ---------------------------------------------------------------------------
# classify_query_node 降级：llm_unavailable 标记 + current_agent + 不升级
# ---------------------------------------------------------------------------

class _FailingClassifyTool:
    """模拟 classify_query 工具（StructuredTool 需 .invoke 接口），恒抛 403"""

    def invoke(self, args, **kwargs):
        raise LLMServiceUnavailable("LLM 通道返回 403（额度耗尽）", status_code=403)


def _run_classify_node(state_extra=None):
    """构造最小 state 并调用 classify_query_node，返回其产出的 state"""
    import nodes.classify as _nc

    state = {"customer_query": "手机多少钱"}
    if state_extra:
        state.update(state_extra)

    # 替换模块级 classify_query 引用为故障 mock（node 内调 classify_query.invoke）
    # 2026-08-18 拆包：patch 定义模块（壳 re-export 是绑定拷贝）
    orig = _nc.classify_query
    _nc.classify_query = _FailingClassifyTool()
    try:
        return _nc.classify_query_node(state)
    finally:
        _nc.classify_query = orig


def test_degrade_sets_llm_unavailable_and_no_escalation():
    """通道故障降级：设 llm_unavailable=True、current_agent=智能客服、不升级、友好提示"""
    state = _run_classify_node()
    assert state.get("llm_unavailable") is True, "通道故障必须打 llm_unavailable 标记"
    assert state.get("current_agent") == "智能客服", "final_response 依赖 current_agent"
    assert state.get("escalated") is False, "通道故障不应升级人工"
    assert "暂时不可用" in state.get("response", ""), "应为友好降级提示"
    assert state.get("query_type") == "general_inquiry"


def test_normal_classify_clears_llm_unavailable(monkeypatch):
    """正常分类（非通道故障）必须清 llm_unavailable，避免跨轮残留误跳 final_response"""
    import nodes.classify as _nc

    class _GoodClassifyTool:
        def invoke(self, args, **kwargs):
            return '{"label": "product_info", "confidence": 0.95, "complexity": "simple"}'

    state = {"customer_query": "手机多少钱", "llm_unavailable": True}  # 模拟上一轮残留
    orig = _nc.classify_query
    _nc.classify_query = _GoodClassifyTool()
    try:
        out = _nc.classify_query_node(state)
    finally:
        _nc.classify_query = orig
    assert out.get("llm_unavailable") is False, "正常分类必须清残留标记"
    assert out.get("query_type") == "product_info"