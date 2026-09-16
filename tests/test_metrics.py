"""测试：metrics 运行时指标聚合（第二波可观测性，2026-09-16）。

风险对照（每个测试对应具名风险）：
- 聚合数学错误（P50/升级率/token 口径算错 → 告警误判）→ 精确值断言
- 窗口过滤失效（旧数据污染当前窗口）→ 旧 run 排除断言
- 告警漏报/误报（超阈不报、小样本误报）→ 触发与豁免双向断言
- 熔断 OPEN 无告警（通道故障静默）→ 状态量告警断言
- llm_slot 计数不配对（泄漏 → 饱和度永久虚高）→ 嵌套/异常路径配对断言
- trace.db 不存在时端点 500（可观测性反成故障面）→ 零值结构断言
"""

import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trace_store


def _ts(minutes_ago: float = 0) -> str:
    return (datetime.now() - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%d %H:%M:%S")


def _seed_run(run_id, ts, total_ms=None, prompt=0, completion=0, decision=None, steps=None):
    """经 trace_store 真实写入路径种子数据（同时验证 schema 契约）。"""
    trace_store.start_run(run_id, "thread-x", "测试问题", ts)
    if total_ms is not None:
        trace_store.finish_run(run_id, total_ms, prompt, completion, decision or {})
    for node, dur, detail in (steps or []):
        trace_store.add_step(run_id, node, dur, 0, detail)


@pytest.fixture(autouse=True)
def _metrics_env(tmp_path, monkeypatch):
    """隔离：独立 trace.db + 熔断器状态前后复位（配对清理，AGENTS.md 纪律）。"""
    from llm.resilience import _circuit_breaker

    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "trace.db"))

    def _reset_breaker():
        _circuit_breaker._state = "CLOSED"
        _circuit_breaker._failure_count = 0
        _circuit_breaker._success_count = 0

    _reset_breaker()
    yield
    _reset_breaker()


# ---------------------------------------------------------------------------
# 聚合数学精确性
# ---------------------------------------------------------------------------

def test_collect_aggregates_exact_values():
    """10 个已知 run → P50/P95（nearest-rank）、升级率、错误率、token 汇总全部精确对拍。"""
    import metrics

    for i in range(1, 11):
        decision = {
            "query_type": "complaint" if i >= 9 else "product_info",
            "confidence": 0.9,
            "escalated": i >= 9,
            "escalation_reason": "投诉诉求" if i >= 9 else "",
            "agent": "投诉处理" if i >= 9 else "产品专家",
        }
        steps = [("classify_query", 150, None), ("final_response", 100, None)]
        if i == 3:
            steps.append(("agents", 50, {"error": "boom"}))
        _seed_run(f"r{i}", _ts(1), total_ms=1000 * i, prompt=100 * i, completion=10 * i,
                  decision=decision, steps=steps)

    m = metrics.collect(window_min=60)

    assert m["traffic"]["runs"] == 10
    assert m["traffic"]["incomplete_runs"] == 0
    assert m["traffic"]["by_query_type"] == {"product_info": 8, "complaint": 2}
    # durations 排序后 [1000..10000]：p50 nearest-rank k=5 → 5000ms；p95 k=10 → 10000ms
    assert m["latency"]["p50_s"] == 5.0
    assert m["latency"]["p95_s"] == 10.0
    assert m["latency"]["samples"] == 10
    assert m["errors"]["runs_with_error"] == 1
    assert m["errors"]["error_rate"] == 0.1
    assert m["escalation"]["escalated"] == 2
    assert m["escalation"]["rate"] == 0.2
    assert m["escalation"]["reasons"] == {"投诉诉求": 2}
    assert m["cost"]["prompt_tokens"] == 5500
    assert m["cost"]["completion_tokens"] == 550
    assert m["cost"]["avg_tokens_per_run"] == 605.0
    assert m["cost"]["est_yuan"] == round(5500 / 1e6 * 0.15 + 550 / 1e6 * 1.5, 4)
    # 最慢节点按窗口聚合：classify_query 每run 150ms（平均最高者之一）
    top_nodes = {n["node"] for n in m["nodes"]}
    assert {"classify_query", "final_response", "agents"} <= top_nodes
    # 该种子含 1 个错误步（10% > 5% 阈值）→ 恰好一条错误率告警，其余指标健康不告警
    assert [a["metric"] for a in m["alerts"]] == ["error_rate"]


def test_incomplete_runs_counted_separately():
    """start 后未 finish 的 run（进程崩溃）计入 incomplete，不进延迟分位。"""
    import metrics

    _seed_run("done", _ts(1), total_ms=3000, prompt=100, completion=10,
              decision={"query_type": "product_info", "escalated": False})
    trace_store.start_run("crashed", "thread-x", "未完成", _ts(1))  # 无 finish_run

    m = metrics.collect(window_min=60)
    assert m["traffic"]["runs"] == 2
    assert m["traffic"]["incomplete_runs"] == 1
    assert m["latency"]["samples"] == 1
    assert m["latency"]["p50_s"] == 3.0


# ---------------------------------------------------------------------------
# 窗口过滤
# ---------------------------------------------------------------------------

def test_window_excludes_old_runs():
    """窗口外的旧 run（含慢/升级）不计入当前窗口任何指标。"""
    import metrics

    _seed_run("old-slow", _ts(120), total_ms=60000, prompt=1, completion=1,
              decision={"query_type": "complaint", "escalated": True,
                        "escalation_reason": "旧数据"})
    for i in range(5):
        _seed_run(f"fresh{i}", _ts(1), total_ms=2000, prompt=10, completion=1,
                  decision={"query_type": "product_info", "escalated": False})

    m = metrics.collect(window_min=60)
    assert m["traffic"]["runs"] == 5
    assert m["latency"]["p50_s"] == 2.0
    assert m["escalation"]["escalated"] == 0
    assert m["alerts"] == []


# ---------------------------------------------------------------------------
# 告警触发与豁免
# ---------------------------------------------------------------------------

def test_alert_p50_triggers(caplog):
    """窗口 P50 超验收线 6s 且样本足 → WARN 告警 + 结构化日志。"""
    import metrics

    for i in range(6):
        _seed_run(f"slow{i}", _ts(1), total_ms=7000,
                  decision={"query_type": "product_info", "escalated": False})

    with caplog.at_level("WARNING"):
        m = metrics.collect(window_min=60)

    triggered = [a for a in m["alerts"] if a["metric"] == "latency_p50_s"]
    assert triggered, "P50=7.0s > 6.0s 且 n=6 应触发告警"
    assert any("[metrics-alert] latency_p50_s" in r.message for r in caplog.records)


def test_alert_error_rate_triggers():
    """错误率超 5% 阈值 → 告警（错误步归属 run 级统计）。"""
    import metrics

    for i in range(10):
        steps = [("agents", 50, {"error": "x"})] if i == 0 else None
        _seed_run(f"e{i}", _ts(1), total_ms=1000,
                  decision={"query_type": "product_info", "escalated": False}, steps=steps)

    m = metrics.collect(window_min=60)
    assert any(a["metric"] == "error_rate" for a in m["alerts"]), "错误率 10% > 5% 应告警"


def test_small_samples_skip_threshold_alerts():
    """样本 < 最小样本量（5）→ 即使 P50 超线也不告警（防小样本误报）。"""
    import metrics

    for i in range(3):
        _seed_run(f"few{i}", _ts(1), total_ms=9000,
                  decision={"query_type": "product_info", "escalated": False})

    m = metrics.collect(window_min=60)
    assert m["traffic"]["runs"] == 3
    assert m["alerts"] == [], "n=3 < 5 不应触发阈值告警"


def test_circuit_open_alerts_without_samples():
    """熔断 OPEN 是状态量：不受样本量约束，无 run 也必须告警。"""
    import metrics
    from llm.resilience import _circuit_breaker

    _circuit_breaker._state = "OPEN"
    try:
        m = metrics.collect(window_min=60)
        assert any(a["metric"] == "circuit_state" for a in m["alerts"])
    finally:
        _circuit_breaker._state = "CLOSED"


# ---------------------------------------------------------------------------
# 通道观测（llm_slot 在途计量）
# ---------------------------------------------------------------------------

def test_llm_slot_inflight_paired():
    """llm_slot 进入/退出/嵌套/异常路径计数严格配对（泄漏=饱和度永久虚高）。"""
    from llm.resilience import get_channel_status, llm_slot

    assert get_channel_status()["llm_requests_inflight"] == 0
    with llm_slot():
        assert get_channel_status()["llm_requests_inflight"] == 1
        with llm_slot():
            assert get_channel_status()["llm_requests_inflight"] == 2
        assert get_channel_status()["llm_requests_inflight"] == 1
    assert get_channel_status()["llm_requests_inflight"] == 0

    with pytest.raises(RuntimeError):
        with llm_slot():
            raise RuntimeError("异常路径也必须配对回零")
    assert get_channel_status()["llm_requests_inflight"] == 0


def test_channel_status_shape():
    """通道快照四要素（熔断态/失败计数/在途/上限）齐备，供 metrics 消费。"""
    from llm.resilience import get_channel_status

    s = get_channel_status()
    assert set(s) == {"circuit_state", "circuit_failure_count",
                      "llm_requests_inflight", "max_concurrency"}
    assert s["circuit_state"] in ("CLOSED", "OPEN", "HALF_OPEN")
    assert s["max_concurrency"] >= 1


# ---------------------------------------------------------------------------
# 端点与边界
# ---------------------------------------------------------------------------

def test_missing_db_returns_zero_structure():
    """trace.db 不存在 → 恒定形状的零值结构（端点不 500）。"""
    import metrics

    monkey_env = os.environ.copy()
    os.environ["TRACE_DB_PATH"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nonexistent-metrics.db")
    try:
        m = metrics.collect(window_min=60)
    finally:
        os.environ.clear()
        os.environ.update(monkey_env)

    assert m["traffic"]["runs"] == 0
    assert m["latency"]["p50_s"] is None
    assert m["alerts"] == []
    assert m["channel"]["circuit_state"] == "CLOSED"


def test_metrics_endpoint_returns_json():
    """GET /api/metrics → 200 + 四大信号分节齐备（Flask 进程内直连，不发请求）。"""
    os.environ.setdefault("FLASK_SECRET_KEY", "test-metrics-key")
    import web_app

    _seed_run("ep1", _ts(1), total_ms=2500,
              decision={"query_type": "product_info", "escalated": False})

    client = web_app.app.test_client()
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.get_json()
    for section in ("traffic", "latency", "errors", "escalation", "cost", "channel", "alerts"):
        assert section in body, f"端点响应缺 {section} 分节"
    assert body["traffic"]["runs"] == 1
    assert body["latency"]["p50_s"] == 2.5
