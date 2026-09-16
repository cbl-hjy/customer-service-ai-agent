"""V18 全链路追踪（2026-08-14）：节点包装器，零侵入业务代码。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

import threading
import time
from datetime import datetime

from llm.resilience import _TOKEN_LOCK, _TOKEN_USAGE
from llm import stream_sink

# 当前 run_id（thread-local：多线程并发下互不串扰；不污染 state/checkpointer）
_trace_local = threading.local()


def _trace_snapshot() -> tuple:
    """token 计数器快照（wrapper 进入时取，退出时算差值 = 本节点 token 消耗）。"""
    with _TOKEN_LOCK:
        return _TOKEN_USAGE["prompt_tokens"], _TOKEN_USAGE["completion_tokens"]


def _traced(node_name: str, fn):
    """节点包装器：记录执行耗时 + token 增量 + 决策字段，写入 trace 表。

    零侵入：不修改 fn 内部逻辑；LangGraph 节点是 state→state 函数，包装后行为不变。
    - run 生命周期：classify_query（入口）首次进入时 start_run + 生成 run_id；
      final_response（出口）执行后 finish_run
    - 异常：记录 error 步后重新抛出（不吞异常，LangGraph 正常传播）
    """
    def wrapped(state):
        from trace_store import add_step, finish_run, start_run

        # token 快照 + 计时
        p0, c0 = _trace_snapshot()
        t0 = time.perf_counter()
        # 进入时若已有 run_id 且是入口节点，先初始化 run（幂等）
        run_id = getattr(_trace_local, "run_id", None)
        if node_name == "classify_query" and run_id is None:
            import uuid
            run_id = f"{uuid.uuid4().hex[:12]}"
            _trace_local.run_id = run_id
            _trace_local.run_start = time.perf_counter()  # V18 修：run 总时长起点
            _trace_local.run_start_tokens = (p0, c0)  # V18 修：run token 累计起点
            thread_id = str(state.get("session_id") or state.get("customer_query") or "unknown")
            start_run(run_id, thread_id, str(state.get("customer_query", "")),
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        # P1 阶段流：节点开始时推送 stage（web 流式请求时前端可见处理进度；
        # 非流式路径 push 内部发现无 sink 直接跳过，零开销零行为变化）
        stream_sink.push("stage", {"node": node_name})
        try:
            result = fn(state)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            p1, c1 = _trace_snapshot()
            # 提取决策字段（result 或 state——节点可能返回新 dict）
            src = result if isinstance(result, dict) else state
            detail = {
                "query_type": src.get("query_type", ""),
                "confidence": src.get("query_confidence", 0.0),
                "escalated": bool(src.get("escalated", False)),
                "escalation_reason": src.get("escalation_reason", ""),
                "agent": src.get("current_agent", ""),
            }
            if run_id:
                add_step(run_id, node_name, round(elapsed_ms, 2), (p1 - p0) + (c1 - c0), detail)
                # 出口节点：回填 run 汇总（总耗时/token = run 起点到完成）+ 清理 run_id
                if node_name == "final_response":
                    run_start = getattr(_trace_local, "run_start", t0)
                    total_ms = (time.perf_counter() - run_start) * 1000
                    ps, cs = getattr(_trace_local, "run_start_tokens", (p0, c0))
                    finish_run(run_id, round(total_ms, 2), p1 - ps, c1 - cs, detail)
                    _trace_local.run_id = None
                    _trace_local.run_start = None
                    _trace_local.run_start_tokens = None
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if run_id:
                add_step(run_id, node_name, round(elapsed_ms, 2), 0, {"error": str(e)[:200]})
                # V18 修：异常时也结束 run（防 thread-local run_id 残留污染下个请求）
                finish_run(run_id, round(elapsed_ms, 2), 0, 0, {"error": str(e)[:200]})
                _trace_local.run_id = None
                _trace_local.run_start = None
                _trace_local.run_start_tokens = None
            raise
    return wrapped
