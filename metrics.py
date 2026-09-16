"""运行时指标聚合（第二波可观测性，2026-09-16）：四大黄金信号 + LLM 特有维度。

设计（调研定标：Google SRE 四大黄金信号 + RED 方法 + devops.com LLM 生产实践）：
    从 trace.db 只读聚合时间窗口内指标，经 /api/metrics 暴露；超阈产出结构化
    WARN 日志 + 响应内 alerts 数组。不接 Prometheus/OTel（单机部署，零依赖纪律）。

四大黄金信号映射：
    Latency   → p50/p95（成功/失败分列；分位数不用平均值——平均值掩盖长尾）
    Traffic   → 窗口请求数 + 决策分布（query_type）
    Errors    → 错误 run 数/比率（trace_steps.detail.error 非空即错误步）
    Saturation→ 熔断器状态 + LLM 在途请求 / 并发上限（resilience.get_channel_status）

LLM 特有维度：升级率+原因分布（fail-safe 健康信号，KB 缺口/分类漂移 → 升级飙升）、
    窗口 token/成本（额度耗尽类事故的前置信号，本项目 glm-5.2 额度事件教训）、最慢节点 Top5。

非显然不变量：
    - 只读：绝不写 trace.db（短连接 + mode=ro）；聚合可容忍秒级陈旧
    - 隐私：输出只含计数/分位/聚合，不含任何用户查询原文（对齐 V9 日志纪律）
    - 告警最小样本量：窗口 run 数 < METRICS_ALERT_MIN_SAMPLES 时跳过阈值判断（防小样本误报）
"""

import logging
import math
import os
import sqlite3
from datetime import datetime, timedelta

from config import (
    METRICS_ALERT_ERROR_RATE,
    METRICS_ALERT_MIN_SAMPLES,
    METRICS_ALERT_P50_S,
    METRICS_TOKEN_IN_PRICE_PER_M,
    METRICS_TOKEN_OUT_PRICE_PER_M,
    METRICS_WINDOW_MIN,
)

logger = logging.getLogger(__name__)

# 与 trace_store 相同的默认路径解析（只读方不复制建表逻辑，文件不存在即空指标）
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trace.db")


def _percentile(sorted_vals: list, p: float):
    """最近邻分位数（nearest-rank）：空列表返回 None。

    选 nearest-rank 而非线性插值：可解释、可手算对拍（测试断言精确值）。
    """
    if not sorted_vals:
        return None
    k = max(1, math.ceil(p / 100 * len(sorted_vals)))
    return sorted_vals[k - 1]


def _connect_ro() -> sqlite3.Connection:
    """只读连接 trace.db（mode=ro：聚合方即使误写也被 SQLite 拒绝）。"""
    db_path = os.environ.get("TRACE_DB_PATH", _DB_PATH)
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _empty_result(window_min: int) -> dict:
    """trace.db 不存在/无数据时的零值结构（端点恒定形状，消费方免判空）。"""
    return {
        "window_minutes": window_min,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "traffic": {"runs": 0, "incomplete_runs": 0, "by_query_type": {}},
        "latency": {"p50_s": None, "p95_s": None, "error_runs_p50_s": None, "samples": 0},
        "errors": {"runs_with_error": 0, "error_rate": 0.0},
        "escalation": {"escalated": 0, "rate": 0.0, "reasons": {}},
        "cost": {"prompt_tokens": 0, "completion_tokens": 0, "est_yuan": 0.0,
                 "avg_tokens_per_run": 0.0,
                 "pricing": {"in_per_m": METRICS_TOKEN_IN_PRICE_PER_M,
                             "out_per_m": METRICS_TOKEN_OUT_PRICE_PER_M}},
        "nodes": [],
        "channel": {},
        "alerts": [],
    }


def collect(window_min: int = METRICS_WINDOW_MIN) -> dict:
    """聚合窗口内指标 + 评估告警。只读，无副作用。"""
    from llm.resilience import get_channel_status

    result = _empty_result(window_min)
    result["channel"] = get_channel_status()

    cutoff = (datetime.now() - timedelta(minutes=window_min)).strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = _connect_ro()
    except FileNotFoundError:
        _evaluate_alerts(result)
        return result

    try:
        rows = conn.execute(
            "SELECT run_id, total_ms, prompt_tokens, completion_tokens, decision "
            "FROM trace_runs WHERE ts >= ?",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    result["traffic"]["runs"] = len(rows)
    if not rows:
        _evaluate_alerts(result)
        return result

    # ---- Latency / Traffic(决策分布) / Escalation / Cost ----
    durations, error_run_ids, by_type, reasons = [], set(), {}, {}
    escalated = decided = incomplete = 0
    prompt_tok = completion_tok = 0
    for r in rows:
        if r["total_ms"] is not None:
            durations.append(r["total_ms"])
        else:
            incomplete += 1  # start_run 后未 finish（进程崩溃/断连未落库）
        prompt_tok += r["prompt_tokens"] or 0
        completion_tok += r["completion_tokens"] or 0
        if r["decision"]:
            import json
            d = json.loads(r["decision"])
            decided += 1
            by_type[d.get("query_type", "unknown")] = by_type.get(d.get("query_type", "unknown"), 0) + 1
            if d.get("escalated"):
                escalated += 1
                reason = d.get("escalation_reason") or "未注明"
                reasons[reason] = reasons.get(reason, 0) + 1

    # ---- Errors（错误步 run 归属：单独一次轻量查询）----
    try:
        conn = _connect_ro()
        try:
            err_rows = conn.execute(
                "SELECT DISTINCT run_id FROM trace_steps "
                "WHERE json_extract(detail,'$.error') IS NOT NULL AND run_id IN "
                f"({','.join('?' * len(rows))})",
                [r["run_id"] for r in rows],
            ).fetchall()
            error_run_ids = {e["run_id"] for e in err_rows}
        finally:
            conn.close()
    except sqlite3.OperationalError:
        # json1 不可用（非常规构建）：降级 LIKE 匹配，宁可粗不可断
        try:
            conn = _connect_ro()
            try:
                err_rows = conn.execute(
                    "SELECT DISTINCT run_id FROM trace_steps "
                    "WHERE detail LIKE '%\"error\"%' AND run_id IN "
                    f"({','.join('?' * len(rows))})",
                    [r["run_id"] for r in rows],
                ).fetchall()
                error_run_ids = {e["run_id"] for e in err_rows}
            finally:
                conn.close()
        except sqlite3.OperationalError:
            error_run_ids = set()

    # ---- 最慢节点 Top5（窗口内 run 的 steps 聚合）----
    try:
        conn = _connect_ro()
        try:
            node_rows = conn.execute(
                "SELECT s.node_name AS node, AVG(s.duration_ms) AS avg_ms, COUNT(*) AS cnt "
                "FROM trace_steps s JOIN trace_runs r ON s.run_id = r.run_id "
                "WHERE r.ts >= ? GROUP BY s.node_name ORDER BY avg_ms DESC LIMIT 5",
                (cutoff,),
            ).fetchall()
            result["nodes"] = [
                {"node": n["node"], "avg_ms": round(n["avg_ms"], 1), "count": n["cnt"]}
                for n in node_rows
            ]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        pass

    # ---- 回填聚合结果 ----
    durations.sort()
    result["traffic"]["incomplete_runs"] = incomplete
    result["traffic"]["by_query_type"] = by_type
    result["latency"]["samples"] = len(durations)
    result["latency"]["p50_s"] = round(_percentile(durations, 50) / 1000, 2) if durations else None
    result["latency"]["p95_s"] = round(_percentile(durations, 95) / 1000, 2) if durations else None
    result["errors"]["runs_with_error"] = len(error_run_ids)
    result["errors"]["error_rate"] = round(len(error_run_ids) / len(rows), 4)
    result["escalation"]["escalated"] = escalated
    result["escalation"]["rate"] = round(escalated / decided, 4) if decided else 0.0
    result["escalation"]["reasons"] = reasons
    result["cost"]["prompt_tokens"] = prompt_tok
    result["cost"]["completion_tokens"] = completion_tok
    result["cost"]["est_yuan"] = round(
        prompt_tok / 1_000_000 * METRICS_TOKEN_IN_PRICE_PER_M
        + completion_tok / 1_000_000 * METRICS_TOKEN_OUT_PRICE_PER_M, 4)
    result["cost"]["avg_tokens_per_run"] = round(
        (prompt_tok + completion_tok) / len(rows), 1) if rows else 0.0

    _evaluate_alerts(result)
    return result


def _evaluate_alerts(result: dict) -> None:
    """阈值评估：超阈 → WARN 日志 + alerts 数组。小样本跳过（防误报）。

    熔断 OPEN 无条件告警（状态量，不受样本量约束）。
    """
    alerts = []

    if result["channel"].get("circuit_state") == "OPEN":
        alerts.append({
            "level": "WARN", "metric": "circuit_state",
            "message": "LLM 通道熔断 OPEN：请求被快速失败，检查 API 额度/网络后等待冷却",
        })

    n = result["traffic"]["runs"]
    if n >= METRICS_ALERT_MIN_SAMPLES:
        p50 = result["latency"]["p50_s"]
        if p50 is not None and p50 > METRICS_ALERT_P50_S:
            alerts.append({
                "level": "WARN", "metric": "latency_p50_s",
                "message": f"窗口 P50={p50}s 超过验收线 {METRICS_ALERT_P50_S}s（n={n}）",
            })
        if result["errors"]["error_rate"] > METRICS_ALERT_ERROR_RATE:
            alerts.append({
                "level": "WARN", "metric": "error_rate",
                "message": f"错误率 {result['errors']['error_rate']:.1%} 超过阈值 "
                           f"{METRICS_ALERT_ERROR_RATE:.0%}（{result['errors']['runs_with_error']}/{n}）",
            })

    for a in alerts:
        logger.warning(f"[metrics-alert] {a['metric']}: {a['message']}")
    result["alerts"] = alerts
