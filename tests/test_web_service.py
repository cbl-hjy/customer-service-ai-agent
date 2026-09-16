"""测试：web 服务层进程内适配（fork B 方案）——纯函数解析逻辑，不触发 LLM。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_web_service import (
    conversation_history_from_state_data,
    last_user_question_from_history,
    _message_count_from_state_data,
    _normalize_created_at,
)


def test_conversation_from_persisted_dialogue():
    """fork 权威记录 persisted_dialogue → 对话列表"""
    state = {
        "values": {
            "persisted_dialogue": [
                {"content": "手机多少钱", "is_user": True, "timestamp": "2026-08-13 13:00:00"},
                {"content": "您好，价格区间1999-5999。", "is_user": False, "timestamp": "2026-08-13 13:00:01"},
            ],
            "conversation_history": [],  # 本 fork 可为空
        }
    }
    hist = conversation_history_from_state_data(state)
    assert len(hist) == 2
    assert hist[0]["is_user"] is True and hist[0]["role"] == "user"
    assert hist[1]["is_user"] is False and hist[1]["role"] == "assistant"
    assert hist[1]["content"] == "您好，价格区间1999-5999。"


def test_conversation_dedup_response_prefix():
    """助手正文已在轮次里，不再追加 values.response（避免双线气泡）"""
    state = {
        "values": {
            "persisted_dialogue": [
                {"content": "退款怎么操作", "is_user": True},
                {"content": "您可以自助退款。", "is_user": False},
            ],
            "response": "您可以自助退款。",
        }
    }
    hist = conversation_history_from_state_data(state)
    assert len(hist) == 2  # 不重复追加 response


def test_message_count_uses_persisted_dialogue():
    """计数优先 persisted_dialogue（fork 权威记录）"""
    state = {
        "values": {
            "persisted_dialogue": [{"content": "a", "is_user": True}, {"content": "b", "is_user": False}],
            "conversation_history": [],
        }
    }
    assert _message_count_from_state_data(state) == 2


def test_message_count_fallback_conversation_history():
    """无 persisted_dialogue 时回退 conversation_history"""
    state = {
        "values": {
            "conversation_history": [{"content": "a", "is_user": True}],
        }
    }
    assert _message_count_from_state_data(state) == 1


def test_last_user_question():
    hist = [
        {"content": "能介绍下吗", "is_user": True},
        {"content": "请问具体型号是？", "is_user": True},
        {"content": "这是回复", "is_user": False},
    ]
    assert last_user_question_from_history(hist) == "请问具体型号是？"


def test_normalize_created_at_iso():
    ts = _normalize_created_at("2026-08-13T05:28:06.113482+00:00")
    assert ts > 1_700_000_000
    assert _normalize_created_at("") > 0
    assert _normalize_created_at(0) > 0


# ---------------------------------------------------------------------------
# 待人工处理工单队列（fetch_escalated_tickets）——纯函数/mock checkpointer，不触发 LLM
# ---------------------------------------------------------------------------

class _FakeTup:
    def __init__(self, tid, values, ts):
        self.config = {"configurable": {"thread_id": tid}}
        self.checkpoint = {"channel_values": values, "ts": ts}


def _cp_with(monkeypatch, tmp_path, tups):
    """V11：fetch 走 SQL 直查（_fetch_latest_checkpoints），mock 该层返回假线程。

    同时隔离 tickets 表（TICKETS_DB_PATH → 临时），避免测试物化污染正式 tickets.db。
    """
    import chat_web_service as cws
    monkeypatch.setenv("TICKETS_DB_PATH", str(tmp_path / "test_tickets.db"))
    fake_threads = [
        {
            "thread_id": t.config["configurable"]["thread_id"],
            "values": t.checkpoint["channel_values"],
            "ts": t.checkpoint["ts"],
        }
        for t in tups
    ]
    monkeypatch.setattr(cws, "_fetch_latest_checkpoints", lambda limit=None, offset=0: fake_threads)


def test_tickets_filters_escalated_only(monkeypatch, tmp_path):
    """只返回 escalated=True 的线程，escalated=False 被过滤"""
    _cp_with(monkeypatch, tmp_path, [
        _FakeTup("t-escalated", {
            "escalated": True,
            "customer_query": "我要投诉物流太慢",
            "escalation_reason": "投诉类问题",
            "escalation_summary": '{"reason": "投诉类问题"}',
            "persisted_dialogue": [
                {"content": "我要投诉物流太慢", "is_user": True, "timestamp": "2026-08-13 13:00:00"},
                {"content": "工单已升级", "is_user": False, "timestamp": "2026-08-13 13:00:01"},
            ],
        }, "2026-08-13T05:00:00+00:00"),
        _FakeTup("t-normal", {
            "escalated": False,
            "customer_query": "手机多少钱",
        }, "2026-08-13T05:01:00+00:00"),
    ])
    from chat_web_service import fetch_escalated_tickets
    tickets, err = fetch_escalated_tickets()
    assert err is None
    assert len(tickets) == 1
    assert tickets[0]["thread_id"] == "t-escalated"
    assert tickets[0]["escalation_reason"] == "投诉类问题"


def test_tickets_last_user_question_from_dialogue(monkeypatch, tmp_path):
    """last_user_question 从对话最后一条用户消息取（而非首条）"""
    _cp_with(monkeypatch, tmp_path, [
        _FakeTup("t-multi", {
            "escalated": True,
            "customer_query": "初始问题",
            "escalation_reason": "复杂问题",
            "escalation_summary": "",
            "persisted_dialogue": [
                {"content": "初始问题", "is_user": True},
                {"content": "回复", "is_user": False},
                {"content": "后续追问", "is_user": True},
            ],
        }, "2026-08-13T05:00:00+00:00"),
    ])
    from chat_web_service import fetch_escalated_tickets
    tickets, err = fetch_escalated_tickets()
    assert err is None
    assert tickets[0]["last_user_question"] == "后续追问"
    assert tickets[0]["message_count"] == 3


def test_tickets_empty_when_no_escalation(monkeypatch, tmp_path):
    """无升级记录 → 空列表，无错误"""
    _cp_with(monkeypatch, tmp_path, [
        _FakeTup("t1", {"escalated": False, "customer_query": "a"}, "2026-08-13T05:00:00+00:00"),
    ])
    from chat_web_service import fetch_escalated_tickets
    tickets, err = fetch_escalated_tickets()
    assert err is None
    assert tickets == []


def test_tickets_sorted_newest_first(monkeypatch, tmp_path):
    """按创建时间倒序（最新升级在前）"""
    _cp_with(monkeypatch, tmp_path, [
        _FakeTup("t-old", {"escalated": True, "escalation_reason": "r", "escalation_summary": ""},
                 "2026-08-13T04:00:00+00:00"),
        _FakeTup("t-new", {"escalated": True, "escalation_reason": "r", "escalation_summary": ""},
                 "2026-08-13T06:00:00+00:00"),
    ])
    from chat_web_service import fetch_escalated_tickets
    tickets, err = fetch_escalated_tickets()
    assert err is None
    assert [t["thread_id"] for t in tickets] == ["t-new", "t-old"]