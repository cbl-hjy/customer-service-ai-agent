"""对话归档存储（V12，2026-08-14）：长会话滚动摘要的完整历史落点。

设计（方案 C：独立归档表）：
    persisted_dialogue 是权威对话记录，但长会话无限 append 导致 checkpoint 序列化
    线性膨胀（每轮 invoke 全量序列化 + DB 无限增长）。V12 引入滚动摘要：
    pd 超 DIALOGUE_CAP 条后，最旧的轮次压缩为一句摘要（存 state.dialogue_summary），
    被压缩的完整原文转存本归档表——web 层渲染时「归档 + pd 尾部」合并，
    保证人工客服查看升级工单历史不丢细节（V10 工单闭环不退化）。

    归档表是独立实体（同 tickets_store 模式）：与会话状态分离，短连接 + 全局锁
    （WSL+NTFS 下 SQLite 并发写有 WAL 锁风险，见 sqlite-wsl-fixes），
    路径可通过环境变量 ARCHIVE_DB_PATH 覆盖（测试隔离，对齐 TICKETS_DB_PATH）。

表结构：
    dialogue_archive(thread_id TEXT, seq INTEGER, content TEXT,
                     is_user INTEGER, timestamp TEXT,
                     PRIMARY KEY(thread_id, seq))
    seq 为归档内全局递增序号（压缩时按原对话顺序分配），保证合并渲染顺序正确。
"""

import os
import sqlite3
import threading
from typing import Dict, Iterable, List, Optional

_ARCHIVE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dialogue_archive.db")

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    db_path = os.environ.get("ARCHIVE_DB_PATH", _ARCHIVE_DB_PATH)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS dialogue_archive (
            thread_id TEXT NOT NULL,
            seq       INTEGER NOT NULL,
            content   TEXT NOT NULL,
            is_user   INTEGER NOT NULL,
            timestamp TEXT,
            PRIMARY KEY (thread_id, seq)
        )"""
    )
    conn.commit()
    return conn


def append_entries(thread_id: str, entries: List[Dict[str, object]]) -> int:
    """批量写入被压缩的轮次（按 entries 顺序分配连续 seq）。

    Args:
        thread_id: 会话线程 ID
        entries: 被压缩的轮次列表（[{content, is_user, timestamp}, ...]），
                 顺序即对话原始顺序（pd 头部 → 归档尾部）。

    Returns:
        写入条数。entries 为空时返回 0（不产生 DB 连接）。
    """
    if not entries:
        return 0
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS m FROM dialogue_archive WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            start = row["m"] + 1
            conn.executemany(
                "INSERT OR REPLACE INTO dialogue_archive (thread_id, seq, content, is_user, timestamp) "
                "VALUES (?,?,?,?,?)",
                [
                    (
                        thread_id,
                        start + i,
                        str(e.get("content", "")),
                        1 if e.get("is_user") else 0,
                        str(e.get("timestamp", "") or ""),
                    )
                    for i, e in enumerate(entries)
                ],
            )
            conn.commit()
            return len(entries)
        finally:
            conn.close()


def fetch_entries(thread_id: str) -> List[Dict[str, object]]:
    """按 seq 升序读回该线程全部归档轮次（web 合并渲染用）。"""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT content, is_user, timestamp FROM dialogue_archive "
                "WHERE thread_id=? ORDER BY seq ASC",
                (thread_id,),
            ).fetchall()
            return [
                {
                    "content": r["content"],
                    "is_user": bool(r["is_user"]),
                    "timestamp": r["timestamp"] or "",
                }
                for r in rows
            ]
        finally:
            conn.close()


def count_entries(thread_id: str) -> int:
    """该线程归档条数（会话列表/工单队列的 message_count 合并用）。"""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM dialogue_archive WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            return int(row["c"])
        finally:
            conn.close()


def count_entries_batch(thread_ids: Iterable[str]) -> Dict[str, int]:
    """批量计数（V12：会话列表/工单队列逐线程合并 message_count，避免 N+1 查询）。"""
    ids = [t for t in thread_ids if t]
    if not ids:
        return {}
    with _lock:
        conn = _connect()
        try:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT thread_id, COUNT(*) AS c FROM dialogue_archive "
                f"WHERE thread_id IN ({placeholders}) GROUP BY thread_id",
                ids,
            ).fetchall()
            return {r["thread_id"]: int(r["c"]) for r in rows}
        finally:
            conn.close()
