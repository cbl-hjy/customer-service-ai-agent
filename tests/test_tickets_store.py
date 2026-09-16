"""V10 工单闭环状态流转测试（2026-08-14）：状态机 + 时间戳 + 非法流转。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import tickets_store


@pytest.fixture
def store(monkeypatch, tmp_path):
    """每个测试用独立临时 DB，隔离状态。"""
    monkeypatch.setenv("TICKETS_DB_PATH", str(tmp_path / "test_tickets.db"))
    yield tickets_store


def test_ensure_creates_pending(store):
    assert store.ensure_ticket("t-1") == tickets_store.STATUS_PENDING
    assert store.get_status("t-1") == tickets_store.STATUS_PENDING


def test_ensure_idempotent_keeps_created_at(store):
    store.ensure_ticket("t-1", "2026-08-14 10:00:00")
    store.ensure_ticket("t-1", "2026-08-14 11:00:00")  # 已存在，created_at 不覆盖
    st = store.batch_status()["t-1"]
    assert st["created_at"] == "2026-08-14 10:00:00"


def test_claim_pending_to_processing(store):
    store.ensure_ticket("t-1")
    ok, status, _ = store.claim_ticket("t-1")
    assert ok and status == tickets_store.STATUS_PROCESSING
    assert store.batch_status()["t-1"]["assigned_at"], "认领应记录 assigned_at"


def test_claim_processing_rejected(store):
    store.ensure_ticket("t-1")
    store.claim_ticket("t-1")
    ok, _, msg = store.claim_ticket("t-1")
    assert not ok and "处理中" in msg


def test_claim_resolved_rejected(store):
    store.ensure_ticket("t-1")
    store.resolve_ticket("t-1")
    ok, _, msg = store.claim_ticket("t-1")
    assert not ok and "重开" in msg


def test_resolve_from_pending_and_processing(store):
    store.ensure_ticket("t-1")
    ok, status, _ = store.resolve_ticket("t-1")
    assert ok and status == tickets_store.STATUS_RESOLVED
    assert store.batch_status()["t-1"]["resolved_at"], "解决应记录 resolved_at"

    store.ensure_ticket("t-2")
    store.claim_ticket("t-2")
    ok, status, _ = store.resolve_ticket("t-2")
    assert ok and status == tickets_store.STATUS_RESOLVED


def test_resolve_resolved_rejected(store):
    store.ensure_ticket("t-1")
    store.resolve_ticket("t-1")
    ok, _, msg = store.resolve_ticket("t-1")
    assert not ok and "已解决" in msg


def test_reopen_resolved_to_pending_clears_timestamps(store):
    store.ensure_ticket("t-1")
    store.claim_ticket("t-1")
    store.resolve_ticket("t-1")
    ok, status, _ = store.reopen_ticket("t-1")
    assert ok and status == tickets_store.STATUS_PENDING
    st = store.batch_status()["t-1"]
    assert not st["assigned_at"] and not st["resolved_at"], "重开应清空闭环时间戳"


def test_reopen_non_resolved_rejected(store):
    store.ensure_ticket("t-1")
    ok, _, msg = store.reopen_ticket("t-1")
    assert not ok and "已解决" in msg


def test_unmaterialized_ticket_rejected(store):
    ok, _, msg = store.claim_ticket("ghost-1")
    assert not ok and "物化" in msg
    assert store.get_status("ghost-1") is None


def test_batch_status_returns_all(store):
    store.ensure_ticket("t-1")
    store.ensure_ticket("t-2")
    st = store.batch_status()
    assert set(st.keys()) == {"t-1", "t-2"}
    assert st["t-1"]["status"] == tickets_store.STATUS_PENDING
