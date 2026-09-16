"""V18 测试：全链路追踪（trace 面板）——自研可观测性，替代 Langfuse。

⚠️ 隔离纪律（2026-08-14 教训）：本测试文件【不设置全局环境变量、不调用 make_graph】——
checkpointer 是进程级单例，先 make_graph 会把它绑定到本文件路径，污染同进程
test_workflow_integration 的真实 API 测试（合跑 7 失败的根因）。因此：
- store 层：纯函数测试（trace_store 自带路径隔离）
- wrapper 采集：直接包装【假节点函数】（wrapper 只依赖 _TOKEN_USAGE + trace_store，
  不依赖 checkpointer/图）
- 真实 invoke 端到端：由独立 ad-hoc 验证脚本承担（独立进程，Windows temp）

覆盖：
- trace_store 存储：start/finish/add_step/fetch（run 汇总 + steps 序列 + 决策 JSON）
- wrapper 采集：包装假节点 → run + steps（耗时/token/决策字段）
- 升级决策链：detail 字段透传
- 异常记录：error 步
- 页面查询函数：fetch_trace_runs / fetch_trace_detail
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# 每个测试独立 trace 库（fixture 内 setenv + 用完即弃，不污染全局）
# 注意：不设 CHECKPOINT_DB_PATH 等——不碰 checkpointer 单例


@pytest.fixture(autouse=True)
def _trace_env(tmp_path):
    os.environ["TRACE_DB_PATH"] = str(tmp_path / "trace.db")
    # 前置清理 thread-local run_id（测试独立性：不依赖其他文件留下干净全局；
    # stream_sse 等包装 classify_query 入口节点时会残留，2026-09-16 定位）
    from trace import _trace_local
    _trace_local.run_id = None
    yield
    os.environ.pop("TRACE_DB_PATH", None)
    _trace_local.run_id = None


class _FakeNode:
    """假 LangGraph 节点：返回带决策字段的 state（测 wrapper 采集，不依赖图）。"""

    def __init__(self, query_type="complaint", escalated=True, tokens=0):
        self.query_type = query_type
        self.escalated = escalated
        self.tokens = tokens

    def __call__(self, state):
        if self.tokens:
            # 模拟 token 消耗（直接改计数器，wrapper 快照差值应捕获）
            import multi_agent_customer_service as m
            with m._TOKEN_LOCK:
                m._TOKEN_USAGE["prompt_tokens"] += self.tokens
        state["query_type"] = self.query_type
        state["escalated"] = self.escalated
        state["escalation_reason"] = "投诉类问题转人工跟进" if self.escalated else ""
        state["current_agent"] = "人工客服" if self.escalated else "账单专家"
        return state


def _base_state():
    return {
        "session_id": "t1",
        "customer_query": "测试问题",
        "query_type": "",
        "query_confidence": 0.0,
        "escalated": False,
        "escalation_reason": "",
        "current_agent": "",
    }


# ---------------------------------------------------------------------------
# trace_store 存储
# ---------------------------------------------------------------------------

def test_store_roundtrip():
    """start → add_step × 2 → finish → fetch 完整回读。"""
    from trace_store import start_run, add_step, finish_run, fetch_run_detail, fetch_runs
    start_run("r1", "t1", "测试问题", "2026-08-14 12:00:00")
    add_step("r1", "classify_query", 10.5, 100, {"query_type": "billing"})
    add_step("r1", "final_response", 5.2, 50, {"escalated": False})
    finish_run("r1", 15.7, 100, 50, {"query_type": "billing", "escalated": False})

    detail = fetch_run_detail("r1")
    assert detail is not None
    assert detail["user_query"] == "测试问题"
    assert detail["total_ms"] == 15.7
    assert detail["prompt_tokens"] == 100 and detail["completion_tokens"] == 50
    assert len(detail["steps"]) == 2
    assert detail["steps"][0]["node_name"] == "classify_query"
    assert detail["steps"][0]["detail"]["query_type"] == "billing"

    runs = fetch_runs()
    assert len(runs) == 1
    assert runs[0]["decision"]["escalated"] is False


def test_store_missing_run_returns_none():
    from trace_store import fetch_run_detail
    assert fetch_run_detail("nonexistent") is None


def test_store_steps_ordered():
    """steps 按写入顺序返回（决策链时间线正确）。"""
    from trace_store import start_run, add_step, finish_run, fetch_run_detail
    start_run("r2", "t1", "q", "2026-08-14 12:00:00")
    for name in ["classify_query", "product_agent", "final_response"]:
        add_step("r2", name, 1.0, 0, {})
    finish_run("r2", 3.0, 0, 0, {})
    detail = fetch_run_detail("r2")
    assert [s["node_name"] for s in detail["steps"]] == ["classify_query", "product_agent", "final_response"]


# ---------------------------------------------------------------------------
# wrapper 采集（直接包装假节点，不依赖图/checkpointer）
# ---------------------------------------------------------------------------

def test_wrapper_records_run_and_steps():
    """包装假节点序列（classify→escalate→final）→ run + 3 steps + 决策字段。"""
    from multi_agent_customer_service import _traced
    from trace_store import fetch_runs, fetch_run_detail

    node = _FakeNode(query_type="complaint", escalated=True)
    wrapped = _traced("classify_query", node)
    st = _base_state()
    wrapped(st)  # 进入 classify：start_run + 生成 run_id

    # 中间节点（复用同一 run_id——thread-local 在 wrapper 内）
    _traced("escalate", lambda s: s)(st)
    _traced("final_response", lambda s: s)(st)  # 出口：finish_run + 清 run_id

    runs = fetch_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run["user_query"] == "测试问题"
    assert run["total_ms"] is not None and run["total_ms"] >= 0
    assert run["decision"]["query_type"] == "complaint"
    assert run["decision"]["escalated"] is True
    assert run["decision"]["escalation_reason"] == "投诉类问题转人工跟进"

    detail = fetch_run_detail(run["run_id"])
    names = [s["node_name"] for s in detail["steps"]]
    assert names == ["classify_query", "escalate", "final_response"], f"决策链异常: {names}"


def test_wrapper_captures_token_delta():
    """包装节点内消耗 token → wrapper 差值捕获（prompt_tokens 增量正确）。"""
    from multi_agent_customer_service import _traced, _TOKEN_USAGE, _TOKEN_LOCK
    from trace_store import fetch_run_detail, fetch_runs

    with _TOKEN_LOCK:
        _TOKEN_USAGE["prompt_tokens"] = 0
        _TOKEN_USAGE["completion_tokens"] = 0

    node = _FakeNode(query_type="billing", escalated=False, tokens=100)
    _traced("classify_query", node)(_base_state())
    _traced("final_response", lambda s: s)(_base_state())

    runs = fetch_runs()
    detail = fetch_run_detail(runs[0]["run_id"])
    # classify 步 token_delta = 100（节点内消耗）
    assert detail["steps"][0]["token_delta"] == 100, f"token 差值捕获失败: {detail['steps'][0]['token_delta']}"
    assert detail["prompt_tokens"] == 100


def test_wrapper_answered_workflow_fields():
    """直接回复工单（escalated=False）→ 决策字段透传。

    注意：三个节点必须共享同一 state 对象（LangGraph 语义——真实图中 state 贯穿全链），
    传新对象会导致 detail 从空 state 提取。
    """
    from multi_agent_customer_service import _traced
    from trace_store import fetch_runs

    node = _FakeNode(query_type="product_info", escalated=False)
    st = _base_state()
    _traced("classify_query", node)(st)
    _traced("product_agent", lambda s: s)(st)
    _traced("final_response", lambda s: s)(st)

    runs = fetch_runs()
    assert runs[0]["decision"]["escalated"] is False
    assert runs[0]["decision"]["query_type"] == "product_info"


def test_wrapper_error_step_recorded():
    """节点抛异常 → error 步记录（不吞异常）。"""
    from multi_agent_customer_service import _traced
    from trace_store import fetch_run_detail, fetch_runs

    def boom(state):
        raise RuntimeError("boom")

    _traced("classify_query", _FakeNode())(_base_state())
    with pytest.raises(RuntimeError):
        _traced("product_agent", boom)(_base_state())

    runs = fetch_runs()
    detail = fetch_run_detail(runs[0]["run_id"])
    assert detail["steps"][-1]["detail"].get("error") == "boom"


# ---------------------------------------------------------------------------
# 页面查询函数
# ---------------------------------------------------------------------------

def test_fetch_trace_functions():
    """chat_web_service 查询函数：列表 + 详情 + 不存在。"""
    from multi_agent_customer_service import _traced
    from chat_web_service import fetch_trace_runs, fetch_trace_detail

    _traced("classify_query", _FakeNode())(_base_state())
    _traced("final_response", lambda s: s)(_base_state())

    runs, err = fetch_trace_runs()
    assert err is None
    assert len(runs) == 1
    run_id = runs[0]["run_id"]
    detail, err2 = fetch_trace_detail(run_id)
    assert err2 is None
    assert detail["run_id"] == run_id
    _, err3 = fetch_trace_detail("no-such-run")
    assert err3 is not None
