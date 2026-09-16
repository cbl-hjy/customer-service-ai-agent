"""V11 SQL 直查测试（2026-08-14）：每线程最新 checkpoint + 分页 + ns 过滤。"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_web_service import _latest_checkpoint_rows


def _make_db(tmp_path, spec):
    """构造与 langgraph checkpoints 同 schema 的假表。

    spec: {thread_id: [(checkpoint_blob, metadata_blob), ...]} 按插入顺序表示多轮。
    """
    db = str(tmp_path / "ck.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB
        )"""
    )
    for tid, rounds in spec.items():
        for i, (cp, md) in enumerate(rounds):
            conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata) "
                "VALUES (?,?,?,?,?,?)",
                (tid, "", f"ck-{tid}-{i}", "json", cp, md),
            )
    conn.commit()
    conn.close()
    return db


def test_latest_per_thread(tmp_path):
    """每线程应取 rowid 最大（最新）的一行。"""
    db = _make_db(tmp_path, {
        "t1": [(b"old-round", b"m1"), (b"new-round", b"m2")],
        "t2": [(b"only-round", b"m3")],
    })
    rows = _latest_checkpoint_rows(db)
    assert len(rows) == 2, "应返回 2 个线程"
    by = {tid: cp for tid, _t, cp, _m in rows}
    assert by["t1"] == b"new-round", "t1 应取最新一轮"
    assert by["t2"] == b"only-round"


def test_limit_offset_pagination(tmp_path):
    """分页：limit/offset 不重叠、能覆盖全部。"""
    db = _make_db(tmp_path, {f"t{i}": [(b"x", b"m")] for i in range(5)})
    page1 = _latest_checkpoint_rows(db, limit=2, offset=0)
    page2 = _latest_checkpoint_rows(db, limit=2, offset=2)
    page3 = _latest_checkpoint_rows(db, limit=2, offset=4)
    assert len(page1) == 2 and len(page2) == 2 and len(page3) == 1
    ids1 = {r[0] for r in page1}
    ids2 = {r[0] for r in page2}
    ids3 = {r[0] for r in page3}
    assert not (ids1 & ids2), "分页不应重叠"
    assert len(ids1 | ids2 | ids3) == 5, "分页应覆盖全部"


def test_ns_filter_ignores_custom_namespaces(tmp_path):
    """checkpoint_ns 非空的线程行不参与主命名空间查询。"""
    db = _make_db(tmp_path, {"t1": [(b"main", b"m")]})
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint) "
        "VALUES ('t1','custom-ns','x',?)",
        (b"custom",),
    )
    conn.commit()
    conn.close()
    rows = _latest_checkpoint_rows(db)
    assert len(rows) == 1, "自定义 ns 的行不应出现在主查询"
    assert rows[0][2] == b"main"


def test_empty_db_returns_empty(tmp_path):
    db = _make_db(tmp_path, {})
    assert _latest_checkpoint_rows(db) == []
