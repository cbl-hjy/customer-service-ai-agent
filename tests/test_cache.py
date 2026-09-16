"""测试：回复缓存（cache.ReplyCache / cached_invoke）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from cache import ReplyCache


def test_normalize_variants_hit_same_key():
    """同一问题的变体（大小写/空格）命中同一缓存 key"""
    c = ReplyCache()
    c.put("  怎么退货  ", {"response": "A"})
    assert c.get("怎么 退货") == {"response": "A"}


def test_get_miss_returns_none():
    c = ReplyCache()
    assert c.get("不存在的问法") is None


def test_lru_eviction():
    c = ReplyCache(max_size=2)
    c.put("q1", {"response": "1"})
    c.put("q2", {"response": "2"})
    c.get("q1")  # q1 被访问 → 移到最近
    c.put("q3", {"response": "3"})  # 淘汰最久未用的 q2
    assert c.get("q2") is None
    assert c.get("q1") == {"response": "1"}
    assert c.get("q3") == {"response": "3"}


def test_thread_safety_basic():
    """多线程并发 get/put 不崩溃（基础冒烟）"""
    import threading

    c = ReplyCache()
    errors = []

    def worker(i):
        try:
            for j in range(50):
                c.put(f"q{i}-{j}", {"response": str(j)})
                c.get(f"q{i}-{j}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"线程安全失败: {errors[:3]}"


def test_override_same_key():
    """相同问题重复回答 → 覆盖旧值（保持最新）"""
    c = ReplyCache()
    c.put("手机多少钱", {"response": "v1"})
    c.put("手机多少钱", {"response": "v2"})
    assert c.get("手机多少钱") == {"response": "v2"}
    assert c.size == 1
