"""工单状态存储（V10，2026-08-14）：人工侧闭环的独立实体。

设计：工单是独立实体，与会话状态（checkpointer channel_values）分离——
人工侧操作（认领/解决/重开）不写 agent 状态机，避免污染；状态存独立 SQLite 表。
首次升级的工单在队列读取时 lazy 物化（ensure_ticket），默认 pending。

状态机：
    pending（待处理）→ processing（处理中）→ resolved（已解决）
    resolved → pending（重开）
    resolve 允许 pending/processing → resolved；claim 只允许 pending → processing。

线程安全：每操作短连接 + 全局锁（WSL+NTFS 下 SQLite 并发写有 WAL 锁风险，见 sqlite-wsl-fixes）。
"""

import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Optional, Tuple

_TICKETS_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tickets.db")

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_RESOLVED = "resolved"

_VALID_STATUS = (STATUS_PENDING, STATUS_PROCESSING, STATUS_RESOLVED)

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _connect() -> sqlite3.Connection:
    db_path = os.environ.get("TICKETS_DB_PATH", _TICKETS_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
            thread_id   TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'pending',
            assigned_at TEXT,
            resolved_at TEXT,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def ensure_ticket(thread_id: str, created_at: Optional[str] = None) -> str:
    """工单首次出现时物化（lazy），返回当前状态；已存在则返回既有状态。"""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT status FROM tickets WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if row is not None:
                return row["status"]
            now = _now()
            conn.execute(
                "INSERT OR IGNORE INTO tickets (thread_id, status, created_at, updated_at) "
                "VALUES (?,?,?,?)",
                (thread_id, STATUS_PENDING, created_at or now, now),
            )
            conn.commit()
            return STATUS_PENDING
        finally:
            conn.close()


def claim_ticket(thread_id: str) -> Tuple[bool, str, str]:
    """认领：pending → processing。resolved 需先重开。"""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT status FROM tickets WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if row is None:
                return False, STATUS_PENDING, "工单不存在（尚未物化，请刷新队列）"
            cur = row["status"]
            if cur == STATUS_RESOLVED:
                return False, cur, "已解决工单不可认领，请先重开"
            if cur == STATUS_PROCESSING:
                return False, cur, "工单已在处理中"
            now = _now()
            conn.execute(
                "UPDATE tickets SET status=?, assigned_at=?, updated_at=? WHERE thread_id=?",
                (STATUS_PROCESSING, now, now, thread_id),
            )
            conn.commit()
            return True, STATUS_PROCESSING, "认领成功"
        finally:
            conn.close()


def resolve_ticket(thread_id: str) -> Tuple[bool, str, str]:
    """标记解决：pending / processing → resolved。"""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT status FROM tickets WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if row is None:
                return False, STATUS_PENDING, "工单不存在（尚未物化，请刷新队列）"
            cur = row["status"]
            if cur == STATUS_RESOLVED:
                return False, cur, "工单已是已解决状态"
            now = _now()
            conn.execute(
                "UPDATE tickets SET status=?, resolved_at=?, updated_at=? WHERE thread_id=?",
                (STATUS_RESOLVED, now, now, thread_id),
            )
            conn.commit()
            return True, STATUS_RESOLVED, "已标记解决"
        finally:
            conn.close()


def reopen_ticket(thread_id: str) -> Tuple[bool, str, str]:
    """重开：resolved → pending（清认领/解决时间戳）。"""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT status FROM tickets WHERE thread_id=?", (thread_id,)
            ).fetchone()
            if row is None:
                return False, STATUS_PENDING, "工单不存在（尚未物化，请刷新队列）"
            cur = row["status"]
            if cur != STATUS_RESOLVED:
                return False, cur, "仅已解决工单可重开"
            now = _now()
            conn.execute(
                "UPDATE tickets SET status=?, assigned_at=NULL, resolved_at=NULL, updated_at=? "
                "WHERE thread_id=?",
                (STATUS_PENDING, now, thread_id),
            )
            conn.commit()
            return True, STATUS_PENDING, "工单已重开"
        finally:
            conn.close()


def get_status(thread_id: str) -> Optional[str]:
    """读取单条工单状态（未物化返回 None）。"""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT status FROM tickets WHERE thread_id=?", (thread_id,)
        ).fetchone()
        return row["status"] if row else None
    finally:
        conn.close()


def batch_status() -> Dict[str, Dict[str, Optional[str]]]:
    """返回全部工单状态映射：{thread_id: {status, created_at, assigned_at, resolved_at}}。"""
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM tickets").fetchall()
        return {
            r["thread_id"]: {
                "status": r["status"],
                "created_at": r["created_at"],
                "assigned_at": r["assigned_at"],
                "resolved_at": r["resolved_at"],
            }
            for r in rows
        }
    finally:
        conn.close()
