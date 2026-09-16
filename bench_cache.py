#!/usr/bin/env python3
"""缓存优化压测（2026-08-12 生产优化①）：相同工单重复请求场景。

对比：无缓存 vs 有缓存（ReplyCache）。
工单池 12 条，发 N 请求（循环 = 大量重复，模拟客服真实重复咨询场景）。
预期：无缓存全走 LLM；有缓存首轮 miss 后全部 hit（秒回）→ 等效吞吐大幅提升。

用法：python bench_cache.py [请求数] [并发]
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

from cache import ReplyCache, cached_invoke
from multi_agent_customer_service import make_graph

TICKETS = [
    "你们最新款手机 X3 Lite 多少钱？和 X2 Max 比哪个性价比高？",
    "我的手机升级系统后一直闪退，重启也没用",
    "昨天买的耳机想退掉，7 天无理由怎么申请？",
    "物流太慢了，等了一周还没到，我要投诉你们！",
    "你们客服几点上班？电话多少？",
    "帮我查一下订单 20260812 的物流进度",
    "帮我写一首关于夏天的诗",
    "退款 10 天没到账，中间催了 3 次没人理，我要找你们经理",
    "平板 Tab Pro 支持触控笔吗？多少钱？",
    "发票抬头可以修改吗？怎么改？",
    "电脑开机黑屏怎么办？",
    "你们的耳机 SoundPro 降噪效果怎么样？",
]

_app = make_graph()


def run_without_cache(n: int, concurrency: int) -> dict:
    pool = [TICKETS[i % len(TICKETS)] for i in range(n)]

    def one(i, q):
        t0 = time.time()
        try:
            _app.invoke({"customer_query": q}, {"configurable": {"thread_id": f"nc-{i}"}})
            return {"ok": True, "dt": time.time() - t0}
        except Exception:  # noqa: BLE001
            return {"ok": False, "dt": time.time() - t0}

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(as_completed([ex.submit(one, i, q) for i, q in enumerate(pool)]))
    total = time.time() - t_start
    return {"total": total, "qps": n / total}


def run_with_cache(n: int, concurrency: int) -> dict:
    cache = ReplyCache()
    pool = [TICKETS[i % len(TICKETS)] for i in range(n)]
    hits = 0

    def one(i, q):
        nonlocal hits
        t0 = time.time()
        try:
            r = cached_invoke(_app, q, f"wc-{i}", cache)
            if r.get("cached"):
                hits += 1
            return {"ok": True, "dt": time.time() - t0}
        except Exception:  # noqa: BLE001
            return {"ok": False, "dt": time.time() - t0}

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(as_completed([ex.submit(one, i, q) for i, q in enumerate(pool)]))
    total = time.time() - t_start
    return {"total": total, "qps": n / total, "hits": hits}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print(f"缓存对比压测：{n} 请求（工单池 12 条循环，模拟重复咨询）| 并发 {concurrency}")
    print("== 场景 A：无缓存（全走 LLM）==")
    a = run_without_cache(n, concurrency)
    print(f"   总耗时 {a['total']:.1f}s | QPS {a['qps']:.2f}")
    print("== 场景 B：有缓存（ReplyCache）==")
    b = run_with_cache(n, concurrency)
    hit_rate = b["hits"] / n
    print(f"   总耗时 {b['total']:.1f}s | QPS {b['qps']:.2f} | 缓存命中率 {hit_rate:.0%}")
    print(f"\n== 提升倍数：QPS {a['qps']:.2f} -> {b['qps']:.2f} = {b['qps'] / a['qps']:.1f}× ==")


if __name__ == "__main__":
    main()
