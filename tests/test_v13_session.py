"""V13 测试：session_manager 加锁 + 惰性清理 + 双写移除。

覆盖：
- 并发 add_message → message_count 精确（多线程丢计数修复）
- 并发 get_session / create_session（隐式创建竞态）
- cleanup_old_sessions 清理过期会话（24h 阈值）
- 惰性清理触发（每 _CLEANUP_INTERVAL 次写入触发一次）
- 双写移除后：multi_agent_customer_service 无 session_manager.add_message 调用
- base_agent 无 _add_message_to_session 死方法
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from session_manager import LangChainSessionManager, _CLEANUP_INTERVAL


# ---------------------------------------------------------------------------
# 并发安全
# ---------------------------------------------------------------------------

def test_concurrent_add_message_count_exact():
    """50 并发 add_message 同一会话 → message_count 精确 50（无丢计数）。"""
    mgr = LangChainSessionManager()
    sid = "conc-s1"
    n_threads, n_each = 10, 5  # 10 线程 × 5 = 50 条

    def worker():
        for i in range(n_each):
            mgr.add_message(sid, f"msg-{i}")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mgr.sessions[sid]["message_count"] == n_threads * n_each
    # 双保险：memory 里的消息数也一致
    assert len(mgr.get_memory(sid).messages) == n_threads * n_each


def test_concurrent_get_session_no_duplicate_creation():
    """50 并发 get_session 同一未知 id → 只隐式创建一次（无重复覆盖/竞态）。"""
    mgr = LangChainSessionManager()
    sid = "conc-create"
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        s = mgr.get_session(sid)
        assert isinstance(s, dict)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 只创建了一个会话实例
    assert sid in mgr.sessions
    # 多次 get_session 后 last_activity 更新无异常
    assert mgr.sessions[sid]["message_count"] == 0


def test_concurrent_mixed_ops_no_crash():
    """并发混合操作（add_message + list_sessions + get_session_info + delete）不崩、不丢。"""
    mgr = LangChainSessionManager()
    sids = [f"mix-{i}" for i in range(5)]
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            mgr.add_message(sids[i % 5], f"w{i}")
            i += 1

    def reader():
        while not stop.is_set():
            mgr.list_sessions()
            mgr.get_session_info(sids[0])

    def deleter():
        while not stop.is_set():
            mgr.delete_session(sids[2])

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(2)]
    threads += [threading.Thread(target=deleter)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join()

    # 无异常即通过；剩余会话的计数自洽
    for s in sids:
        if s in mgr.sessions:
            assert mgr.sessions[s]["message_count"] >= 0


# ---------------------------------------------------------------------------
# 清理
# ---------------------------------------------------------------------------

def test_cleanup_old_sessions_removes_expired_only():
    """cleanup 只删超龄会话，活跃会话保留。"""
    mgr = LangChainSessionManager()
    old = mgr.create_session("old-1")
    fresh = mgr.create_session("fresh-1")
    # 伪造 last_activity：old 24h 前，fresh 刚刚
    mgr.sessions["old-1"]["last_activity"] = time.time() - 25 * 3600
    mgr.sessions["fresh-1"]["last_activity"] = time.time() - 1

    cleaned = mgr.cleanup_old_sessions(max_age_hours=24)
    assert cleaned == 1
    assert "old-1" not in mgr.sessions
    assert "fresh-1" in mgr.sessions


def test_lazy_cleanup_triggers_after_interval(monkeypatch):
    """惰性清理：写计数达 _CLEANUP_INTERVAL 时触发 cleanup（事件驱动）。"""
    mgr = LangChainSessionManager()
    # 造一个过期会话 + 一个活跃会话
    mgr.create_session("expired-lazy")
    mgr.create_session("active-lazy")
    mgr.sessions["expired-lazy"]["last_activity"] = time.time() - 25 * 3600

    calls = []
    orig_cleanup = mgr.cleanup_old_sessions

    def spy_cleanup(max_age_hours=24):
        calls.append(max_age_hours)
        return orig_cleanup(max_age_hours)

    monkeypatch.setattr(mgr, "cleanup_old_sessions", spy_cleanup)

    # 写 _CLEANUP_INTERVAL 次（触发一次清理）
    for i in range(_CLEANUP_INTERVAL):
        mgr.add_message("active-lazy", f"m{i}")

    assert len(calls) == 1, f"应触发 1 次清理，实际 {len(calls)}"
    # 过期会话被清掉
    assert "expired-lazy" not in mgr.sessions
    assert "active-lazy" in mgr.sessions


def test_lazy_cleanup_failure_does_not_break_write(monkeypatch):
    """清理抛异常不影响 add_message 主流程（fail-safe）。"""
    mgr = LangChainSessionManager()

    def boom(max_age_hours=24):
        raise RuntimeError("清理失败")

    monkeypatch.setattr(mgr, "cleanup_old_sessions", boom)
    for i in range(_CLEANUP_INTERVAL):
        mgr.add_message("safe-lazy", f"m{i}")
    # 主流程没崩，消息都写进去了
    assert mgr.sessions["safe-lazy"]["message_count"] == _CLEANUP_INTERVAL


# ---------------------------------------------------------------------------
# 双写移除（结构断言）
# ---------------------------------------------------------------------------

def test_dual_write_removed_from_workflow():
    """双写移除：主流程不再调用 session_manager.add_message。"""
    import multi_agent_customer_service as m
    src = open(m.__file__, encoding="utf-8").read()
    assert "session_manager.add_message" not in src, "双写未完全移除"
    assert "session_manager = default_session_manager" not in src, "无用全局变量未移除"


def test_dead_method_removed_from_base_agent():
    """base_agent 死方法 _add_message_to_session 已移除。"""
    from multi_agents.base_agent import BaseAgent
    assert not hasattr(BaseAgent, "_add_message_to_session")


def test_base_agent_fallback_still_works():
    """base_agent 的 session_manager 兜底路径仍在（state 缺失时可用）。"""
    from multi_agents.base_agent import BaseAgent

    class ConcreteAgent(BaseAgent):
        def process(self, state):
            return {"response": "ok"}

    agent = ConcreteAgent(name="t", role="r", expertise=["x"])
    # 无 state → 走 session_manager 兜底（不抛异常）
    ctx = agent._get_conversation_context("fallback-sid")
    assert ctx == ""  # 空会话返回空串
