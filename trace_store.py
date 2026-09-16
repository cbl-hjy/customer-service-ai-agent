"""全链路追踪存储（V18 trace 面板，2026-08-14）：单次工单决策链可视化。

设计（方案 D：自研 trace，替代 Langfuse——零依赖/中文原生/本地秒开）：
    每次 invoke 的完整决策链记录在两表：
    - trace_runs：一次工单（run）的汇总——用户问题/总耗时/token/决策结果
    - trace_steps：run 内每个节点一步——节点名/耗时/token 增量/决策字段
    采集由 multi_agent_customer_service 的节点包装器 _traced 完成（零侵入业务代码），
    本模块只负责持久化与查询。

    独立实体模式（同 tickets_store/dialogue_archive）：短连接 + 全局锁（WSL+NTFS
    SQLite 并发写风险，见 sqlite-wsl-fixes），路径经 TRACE_DB_PATH 环境变量隔离
    （测试隔离，对齐 TICKETS_DB_PATH / ARCHIVE_DB_PATH）。

表结构：
    trace_runs(run_id TEXT PK, thread_id TEXT, user_query TEXT, ts TEXT,
               total_ms REAL, prompt_tokens INT, completion_tokens INT,
               decision TEXT,          -- decision = JSON（query_type/confidence/escalated/reason/agent）
               response TEXT,          -- T2（2026-09-16）：本轮 agent 回复原文（回流评估数据源）
               tools_used TEXT)        -- T2：本轮工具调用 JSON 数组（订单工具等）
    trace_steps(id INTEGER PK AUTOINCREMENT, run_id TEXT, node_name TEXT,
                duration_ms REAL, token_delta INT, detail TEXT)   -- detail = JSON

保留策略：demo 场景量小，保留全部（不清理）；若需清理后续按 V13 惰性模式加。
历史行 response/tools_used 为 NULL（T2 之前未采集，不可找回）——回流侧兼容。
"""

import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

_TRACE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trace.db")

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    db_path = os.environ.get("TRACE_DB_PATH", _TRACE_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trace_runs (
            run_id          TEXT PRIMARY KEY,
            thread_id       TEXT NOT NULL,
            user_query      TEXT NOT NULL,
            ts              TEXT NOT NULL,
            total_ms        REAL,
            prompt_tokens   INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            decision        TEXT,
            response        TEXT,
            tools_used      TEXT
        )"""
    )
    # T2 迁移（2026-09-16）：旧库补列（幂等）——response/tools_used 为回流评估数据源
    cols = {r[1] for r in conn.execute("PRAGMA table_info(trace_runs)")}
    if "response" not in cols:
        conn.execute("ALTER TABLE trace_runs ADD COLUMN response TEXT")
    if "tools_used" not in cols:
        conn.execute("ALTER TABLE trace_runs ADD COLUMN tools_used TEXT")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS trace_steps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            node_name   TEXT NOT NULL,
            duration_ms REAL,
            token_delta INTEGER DEFAULT 0,
            detail      TEXT
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_steps_run ON trace_steps(run_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON trace_runs(ts)")
    conn.commit()
    return conn


def start_run(run_id: str, thread_id: str, user_query: str, ts: str) -> None:
    """开始一次工单 trace（run 汇总行）。"""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO trace_runs (run_id, thread_id, user_query, ts) VALUES (?,?,?,?)",
                (run_id, thread_id, user_query, ts),
            )
            conn.commit()
        finally:
            conn.close()


def finish_run(run_id: str, total_ms: float, prompt_tokens: int, completion_tokens: int,
               decision: Dict[str, Any], response: str = "", tools_used: Optional[List[str]] = None) -> None:
    """结束一次工单 trace：回填耗时/token/决策结果/回复原文（T2 回流数据源）。"""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "UPDATE trace_runs SET total_ms=?, prompt_tokens=?, completion_tokens=?, decision=?, "
                "response=?, tools_used=? WHERE run_id=?",
                (total_ms, prompt_tokens, completion_tokens, json.dumps(decision, ensure_ascii=False),
                 response or "", json.dumps(tools_used or [], ensure_ascii=False), run_id),
            )
            conn.commit()
        finally:
            conn.close()


def add_step(run_id: str, node_name: str, duration_ms: float, token_delta: int, detail: Optional[Dict[str, Any]] = None) -> None:
    """记录一个节点的执行步。"""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO trace_steps (run_id, node_name, duration_ms, token_delta, detail) VALUES (?,?,?,?,?)",
                (run_id, node_name, duration_ms, token_delta,
                 json.dumps(detail, ensure_ascii=False) if detail else None),
            )
            conn.commit()
        finally:
            conn.close()


def fetch_runs(limit: int = 50) -> List[Dict[str, Any]]:
    """最近 N 次工单 trace（列表页）。按 ts 倒序。"""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT * FROM trace_runs ORDER BY ts DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["decision"] = json.loads(d["decision"]) if d.get("decision") else {}
                out.append(d)
            return out
        finally:
            conn.close()


def fetch_run_detail(run_id: str) -> Optional[Dict[str, Any]]:
    """单次工单完整决策链（详情页）：run 汇总 + 节点步序列。"""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM trace_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return None
            run = dict(row)
            run["decision"] = json.loads(run["decision"]) if run.get("decision") else {}
            steps = conn.execute(
                "SELECT node_name, duration_ms, token_delta, detail FROM trace_steps "
                "WHERE run_id=? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
            run["steps"] = []
            for s in steps:
                d = dict(s)
                d["detail"] = json.loads(d["detail"]) if d.get("detail") else {}
                run["steps"].append(d)
            return run
        finally:
            conn.close()
