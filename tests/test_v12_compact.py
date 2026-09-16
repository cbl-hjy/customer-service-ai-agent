"""V12 测试：对话压缩（滚动摘要 + 归档表 + web 合并渲染）。

覆盖：
- 归档存储读写/顺序/批量计数（dialogue_archive 独立实体）
- 压缩触发边界（超 DIALOGUE_CAP 归档、保留 DIALOGUE_TAIL_KEEP 尾部、摘要生成）
- 压缩后 pd 有界（滚动窗口不无限膨胀）
- web 合并渲染（归档 + 尾部 = 完整历史，顺序正确）
- message_count 合并（pd 尾部 + 归档条数）
- fail-safe：归档失败不压缩不丢历史
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# 每个测试独立归档库（monkeypatch 环境变量需在导入前/导入后重建连接——本模块连接惰性，OK）
_ARCHIVE_DB = "/tmp/hermes-v12-test-archive.db"


@pytest.fixture(autouse=True)
def _isolate_archive_db(tmp_path, monkeypatch):
    db = str(tmp_path / "archive.db")
    monkeypatch.setenv("ARCHIVE_DB_PATH", db)
    yield
    # 清空模块级缓存无（短连接），DB 文件随 tmp_path 清理


from dialogue_archive import (  # noqa: E402
    append_entries,
    fetch_entries,
    count_entries,
    count_entries_batch,
)
from multi_agent_customer_service import (  # noqa: E402
    _compact_dialogue_if_needed,
    _compact_after_append,
    DIALOGUE_CAP,
    DIALOGUE_TAIL_KEEP,
)
from chat_web_service import (  # noqa: E402
    conversation_history_merged,
    _merge_message_count,
)


def _mk_state(thread_id="t1", pd=None, summary=""):
    return {
        "session_id": thread_id,
        "persisted_dialogue": list(pd) if pd else [],
        "dialogue_summary": summary,
    }


# ---------------------------------------------------------------------------
# 归档存储
# ---------------------------------------------------------------------------

def test_archive_append_and_fetch_order():
    """归档写入后按原顺序读回（seq 递增），字段保真。"""
    entries = [
        {"content": "第一条", "is_user": True, "timestamp": "2026-08-14 10:00:00"},
        {"content": "AI 回复", "is_user": False, "timestamp": "2026-08-14 10:00:01"},
        {"content": "第二条", "is_user": True, "timestamp": "2026-08-14 10:00:02"},
    ]
    n = append_entries("t-order", entries)
    assert n == 3
    fetched = fetch_entries("t-order")
    assert [f["content"] for f in fetched] == ["第一条", "AI 回复", "第二条"]
    assert fetched[0]["is_user"] is True
    assert fetched[1]["is_user"] is False


def test_archive_append_empty_is_noop():
    assert append_entries("t-empty", []) == 0
    assert fetch_entries("t-empty") == []


def test_archive_append_sequential_across_calls():
    """多次 append 到同一线程，seq 连续不覆盖（滚动归档累积）。"""
    append_entries("t-seq", [{"content": "a", "is_user": True}])
    append_entries("t-seq", [{"content": "b", "is_user": True}, {"content": "c", "is_user": False}])
    fetched = fetch_entries("t-seq")
    assert [f["content"] for f in fetched] == ["a", "b", "c"]


def test_archive_count_and_batch():
    append_entries("t-c1", [{"content": f"m{i}", "is_user": True} for i in range(3)])
    append_entries("t-c2", [{"content": "x", "is_user": True}])
    assert count_entries("t-c1") == 3
    assert count_entries("t-none") == 0
    assert count_entries_batch(["t-c1", "t-c2", "t-none"]) == {"t-c1": 3, "t-c2": 1}


def test_archive_thread_isolation():
    """不同线程互不串扰。"""
    append_entries("t-iso-1", [{"content": "一", "is_user": True}])
    append_entries("t-iso-2", [{"content": "二", "is_user": True}])
    assert [f["content"] for f in fetch_entries("t-iso-1")] == ["一"]
    assert [f["content"] for f in fetch_entries("t-iso-2")] == ["二"]


# ---------------------------------------------------------------------------
# 压缩逻辑
# ---------------------------------------------------------------------------

def test_compact_below_cap_noop():
    """未超 DIALOGUE_CAP 不压缩、不归档、摘要不变。"""
    state = _mk_state()
    for i in range(DIALOGUE_CAP):
        state["persisted_dialogue"].append({"content": f"m{i}", "is_user": True})
    _compact_dialogue_if_needed(state)
    assert len(state["persisted_dialogue"]) == DIALOGUE_CAP
    assert state["dialogue_summary"] == ""
    assert count_entries("t1") == 0


def test_compact_above_cap_archives_and_keeps_tail():
    """超 cap 后：归档最旧、保留尾部 DIALOGUE_TAIL_KEEP 条、摘要生成。"""
    state = _mk_state()
    total = DIALOGUE_CAP + 10
    for i in range(total):
        state["persisted_dialogue"].append(
            {"content": f"用户{i}", "is_user": True, "timestamp": "t"}
        )
        state["persisted_dialogue"].append(
            {"content": f"回复{i}", "is_user": False, "timestamp": "t"}
        )
    _compact_dialogue_if_needed(state)
    assert len(state["persisted_dialogue"]) == DIALOGUE_TAIL_KEEP
    # 保留的是最新轮次
    assert state["persisted_dialogue"][0]["content"] == f"用户{total - DIALOGUE_TAIL_KEEP//2}"
    # 归档 = 总数 - 保留
    assert count_entries("t1") == total * 2 - DIALOGUE_TAIL_KEEP
    # 摘要含被归档的用户消息（不含 AI 回复）
    assert "用户0" in state["dialogue_summary"]
    assert "回复0" not in state["dialogue_summary"]


def test_compact_rolling_window_bounded():
    """长会话滚动窗口有界：pd 恒 ≤ CAP（超限即压缩回 TAIL_KEEP），归档持续累积。"""
    state = _mk_state()
    for i in range(200):
        state["persisted_dialogue"].append({"content": f"u{i}", "is_user": True, "timestamp": "t"})
        state["persisted_dialogue"].append({"content": f"a{i}", "is_user": False, "timestamp": "t"})
        _compact_dialogue_if_needed(state)
        # 上界：永不超 CAP（滚动窗口的核心不变量）
        assert len(state["persisted_dialogue"]) <= DIALOGUE_CAP
    # 收尾：pd 停在 ≤ CAP 的某个值，其余全部进归档
    assert count_entries("t1") == 400 - len(state["persisted_dialogue"])


def test_compact_after_append_unified_entrypoint():
    """统一接入点：append 后自动触发压缩（无需手工调用）。"""
    state = _mk_state()
    for i in range(DIALOGUE_CAP + 4):
        _compact_after_append(state, {"content": f"m{i}", "is_user": True})
    assert len(state["persisted_dialogue"]) <= DIALOGUE_CAP
    assert count_entries("t1") > 0


def test_compact_no_session_id_skips():
    """无 session_id 时跳过压缩（不崩、不丢 pd）。"""
    state = {"session_id": "", "persisted_dialogue": [{"content": "x", "is_user": True}] * (DIALOGUE_CAP + 5)}
    _compact_dialogue_if_needed(state)
    assert len(state["persisted_dialogue"]) == DIALOGUE_CAP + 5  # 未压缩
    assert state.get("dialogue_summary", "") == ""


def test_compact_archive_failure_failsafe(monkeypatch):
    """归档失败（DB 异常）→ 跳过压缩保留原文（fail-safe 不丢历史）。"""
    import dialogue_archive

    def boom(thread_id, entries):
        raise RuntimeError("DB 损坏")

    monkeypatch.setattr(dialogue_archive, "append_entries", boom)
    state = _mk_state()
    for i in range(DIALOGUE_CAP + 5):
        state["persisted_dialogue"].append({"content": f"m{i}", "is_user": True})
    _compact_dialogue_if_needed(state)
    assert len(state["persisted_dialogue"]) == DIALOGUE_CAP + 5  # 原文保留
    assert state["dialogue_summary"] == ""


def test_compact_summary_truncated(monkeypatch):
    """超长摘要截断到上限（防摘要自身膨胀）。"""
    import nodes.dialogue as nd
    # 模块级常量已求值，monkeypatch 模块属性（自动恢复，不污染同进程其他测试）
    # 2026-08-18 拆包：常量由 nodes.dialogue 持有，patch 定义处
    monkeypatch.setattr(nd, "_DIALOGUE_SUMMARY_MAX_CHARS", 60)
    state = _mk_state()
    for i in range(DIALOGUE_CAP + 2):
        state["persisted_dialogue"].append(
            {"content": "很长的问题内容" * 10, "is_user": True, "timestamp": "t"}
        )
    nd._compact_dialogue_if_needed(state)
    assert len(state["dialogue_summary"]) <= 60 + 1  # 60 + 省略号


# ---------------------------------------------------------------------------
# web 合并渲染
# ---------------------------------------------------------------------------

def _state_data(pd, summary=""):
    return {
        "values": {
            "persisted_dialogue": pd,
            "dialogue_summary": summary,
            "conversation_history": [],
        }
    }


def test_web_merge_archive_plus_tail():
    """会话详情 = 归档 + pd 尾部，顺序完整（人工客服看得到早期对话）。"""
    # 归档 4 条（早期），pd 尾部 2 条（近期）
    append_entries("t-web", [
        {"content": "旧问1", "is_user": True, "timestamp": "t1"},
        {"content": "旧答1", "is_user": False, "timestamp": "t1"},
        {"content": "旧问2", "is_user": True, "timestamp": "t1"},
        {"content": "旧答2", "is_user": False, "timestamp": "t1"},
    ])
    pd_tail = [
        {"content": "新问", "is_user": True, "timestamp": "t2"},
        {"content": "新答", "is_user": False, "timestamp": "t2"},
    ]
    hist = conversation_history_merged("t-web", _state_data(pd_tail))
    assert len(hist) == 6
    assert [h["content"] for h in hist] == ["旧问1", "旧答1", "旧问2", "旧答2", "新问", "新答"]
    assert hist[0]["role"] == "user" and hist[1]["role"] == "assistant"
    # 时间戳保留
    assert hist[0]["timestamp"] == "t1" and hist[4]["timestamp"] == "t2"


def test_web_merge_no_archive_returns_base():
    """无归档时合并函数退化为纯 pd 渲染（不破坏既有行为）。"""
    pd = [{"content": "只有这条", "is_user": True}]
    hist = conversation_history_merged("t-none", _state_data(pd))
    assert len(hist) == 1
    assert hist[0]["content"] == "只有这条"


def test_merge_message_count():
    """消息数 = pd 尾部 + 归档（压缩后计数不缩水）。"""
    assert _merge_message_count("t1", pd_count=16, archive_counts={"t1": 84}) == 100
    assert _merge_message_count("t2", pd_count=5, archive_counts={}) == 5


def test_web_merge_empty_pd_only_archive():
    """pd 尾部为空但归档有历史（极端：全部被归档）→ 只渲染归档。"""
    append_entries("t-empty-pd", [
        {"content": "全归档1", "is_user": True, "timestamp": "t"},
        {"content": "全归档2", "is_user": False, "timestamp": "t"},
    ])
    hist = conversation_history_merged("t-empty-pd", _state_data([]))
    assert [h["content"] for h in hist] == ["全归档1", "全归档2"]
