#!/usr/bin/env python3
"""GPU 检索热路径微剖析：dense/rerank 耗时拆分 + 并发争抢代价 + token 长度分布。

用途：并发容量曲线（bench_20260916）归因后的定点量化——
  1) 单请求 GPU 时间到底花在哪（dense 单句编码 vs rerank 20 对长文本）
  2) 多线程并发调用同一模型单例的争抢惩罚（WDDM 时间片）
  3) rerank 输入 token 长度分布（截断优化空间）

用法：python bench/gpu_profile.py
"""
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_retriever import HybridRetriever


def stats(ts):
    return f"avg={statistics.mean(ts)*1000:.0f}ms p50={sorted(ts)[len(ts)//2]*1000:.0f}ms max={max(ts)*1000:.0f}ms"


def main():
    kb_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "knowledge_base_v2.json",
    )
    h = HybridRetriever(kb_path=kb_path)
    h.warm_up()
    dense = h._get_dense()
    reranker = h._get_reranker()

    domain = next(iter(dense._vectors))
    entries = dense._data[domain][:20]
    q = "耳机降噪效果怎么样"

    # ---- 1) dense 单句编码 ----
    for _ in range(3):
        dense.search(domain, q, top_k=20)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        dense.search(domain, q, top_k=20)
        ts.append(time.perf_counter() - t0)
    print(f"[dense.search 单句编码+CPU余弦] {stats(ts)}")

    # ---- 2) rerank 20 对（生产池规模） ----
    pairs = [(q, dense._entry_text(e)) for e in entries]
    for _ in range(2):
        reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    ts2 = []
    for _ in range(10):
        t0 = time.perf_counter()
        reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        ts2.append(time.perf_counter() - t0)
    print(f"[reranker.predict 20对 fp32] {stats(ts2)}")

    # ---- 3) 并发争抢惩罚：8 线程同时 predict（同一模型单例）----
    n_threads, n_each = 8, 3

    def worker():
        for _ in range(n_each):
            reranker.predict(pairs, batch_size=32, show_progress_bar=False)

    # 串行基线：同总量 24 次
    t0 = time.perf_counter()
    for _ in range(n_threads * n_each):
        reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    serial = time.perf_counter() - t0

    # 并发：8 线程 × 3 次
    barrier = threading.Barrier(n_threads)
    def worker_b():
        barrier.wait()
        for _ in range(n_each):
            reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    threads = [threading.Thread(target=worker_b) for _ in range(n_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    conc = time.perf_counter() - t0
    print(f"[争抢惩罚] 24 次 predict：串行 {serial:.1f}s vs {n_threads}线程并发 {conc:.1f}s（惩罚 {conc/serial:.1f}x）")

    # ---- 4) token 长度分布（截断空间）----
    tok = reranker.tokenizer
    lens = []
    for _, txt in pairs:
        lens.append(len(tok(txt, truncation=False)["input_ids"]))
    lens.sort()
    print(f"[rerank 条目文本 token] n={len(lens)} p50={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
    try:
        print(f"[reranker max_length] {reranker.max_length}")
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
