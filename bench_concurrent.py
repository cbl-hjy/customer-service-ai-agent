#!/usr/bin/env python3
"""
并发压测（W3）：并发工单处理 + 吞吐/延迟分布 + 瓶颈归因。

指标：
- 吞吐 QPS（成功请求数 / 总耗时）
- 延迟分布 P50 / P95 / P99
- 错误率（含 API 限流 429）
- 升级率 / 护栏拒绝率
- 瓶颈归因：无 LLM 编排耗时（降级路径） vs 全链路耗时 → LLM 占比

用法：
    python bench_concurrent.py [并发数] [请求数]     # 默认 10 并发 / 50 请求
    python bench_concurrent.py --overhead            # 只测纯编排开销（不调 LLM）

注意：真实 LLM 调用消耗 API 额度（100 请求约 ¥0.2-1.6）；dashscope 可能限流（429 会记录）。
"""

import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_customer_service import make_graph

# 压测工单池（12 条，含升级/护栏/各业务类）
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

_app = None
_app_lock = threading.Lock()


def _get_app():
    global _app
    if _app is None:
        with _app_lock:
            if _app is None:
                _app = make_graph()
    return _app


def _run_one(index: int, query: str, run_tag: str) -> dict:
    """单条工单（线程安全：独立 thread_id；SqliteSaver 内部有 lock）。

    thread_id 带 run_tag（时间戳）：防跨运行 checkpoint 残留污染（与 eval 清库/
    E2E 冒烟同纪律——同 thread_id 会带上轮对话历史，延迟与行为都失真）。
    """
    t0 = time.time()
    try:
        r = _get_app().invoke(
            {"customer_query": query},
            {"configurable": {"thread_id": f"bench-conc-{run_tag}-{index}"}},
        )
        return {
            "ok": True,
            "dt": time.time() - t0,
            "type": r["query_type"],
            "escalated": r["escalated"],
        }
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "dt": time.time() - t0, "error": str(e)[:100]}


def run_concurrent(concurrency: int, requests: int) -> None:
    app = _get_app()
    app  # noqa: B018 - 预热
    run_tag = time.strftime("%H%M%S")
    # 压测前预热（BENCH_WARMUP=1）：跑 1 条不计时请求，触发 bge-m3/reranker 懒加载，
    # 避免冷启动 23-29s 计入首请求延迟与墙钟（bench_20260915 known_issue 的修正）。
    if os.environ.get("BENCH_WARMUP") == "1":
        _run_one(999, TICKETS[0], run_tag + "w")
        print("[预热] 模型懒加载完成（不计时）")
    pool = list((TICKETS[i % len(TICKETS)]) for i in range(requests))

    t_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_run_one, i, q, run_tag): i for i, q in enumerate(pool)}
        for fut in as_completed(futs):
            results.append(fut.result())
    total = time.time() - t_start

    oks = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    dts = sorted(r["dt"] for r in oks)
    n_escalated = sum(1 for r in oks if r["escalated"])
    n_guardrail = sum(1 for r in oks if r["type"] == "out_of_scope")

    def pct(p):
        if not dts:
            return 0.0
        idx = min(len(dts) - 1, int(len(dts) * p))
        return dts[idx]

    print("===== 并发压测汇总 =====")
    print(f"配置: {concurrency} 并发 × {requests} 请求")
    print(f"总耗时: {total:.1f}s | 吞吐: {len(oks) / total:.2f} QPS")
    print(f"成功: {len(oks)} | 错误: {len(errs)} | 错误率: {len(errs) / requests:.1%}")
    if errs:
        print(f"错误样本: {errs[0]['error']}")
    if dts:
        print(f"延迟: P50={pct(0.50):.2f}s P95={pct(0.95):.2f}s P99={pct(0.99):.2f}s")
    print(f"升级率: {n_escalated}/{len(oks)} | 护栏拒绝: {n_guardrail}/{len(oks)}")
    import config as _cfg
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 模型: {_cfg.OPENAI_MODEL} | MAX_CONCURRENCY={_cfg.MAX_CONCURRENCY}")


def run_overhead() -> None:
    """纯编排开销（无 LLM）：降级路径耗时 ≈ 图编排 + checkpointer + 分类工具"""
    os.environ["DASHSCOPE_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    app = make_graph()
    dts = []
    for i in range(30):
        t0 = time.time()
        app.invoke(
            {"customer_query": "手机多少钱"},
            {"configurable": {"thread_id": f"bench-oh-{i}"}},
        )
        dts.append(time.time() - t0)
    print("===== 纯编排开销（无 LLM，30 次）=====")
    print(f"均值: {statistics.mean(dts):.3f}s | P95: {sorted(dts)[int(len(dts)*0.95)]:.3f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--overhead":
        run_overhead()
    else:
        concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        requests = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_concurrent(concurrency, requests)
