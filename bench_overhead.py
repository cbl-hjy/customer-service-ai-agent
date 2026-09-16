#!/usr/bin/env python3
"""纯编排开销（无 LLM）：降级路径耗时 ≈ 图编排 + checkpointer + 分类工具。

必须在 import multi_agent_customer_service 之前清空 API key（config.py 加载 .env 时不覆盖已设变量）。
结果用于并发压测的瓶颈归因（LLM 占比 = 1 - 编排耗时/全链路耗时）。
"""

import os

os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_customer_service import make_graph  # noqa: E402


def main() -> None:
    app = make_graph()
    dts = []
    for i in range(30):
        t0 = time.time()
        app.invoke(
            {"customer_query": "手机多少钱"},
            {"configurable": {"thread_id": f"bench-oh-{i}"}},
        )
        dts.append(time.time() - t0)
    dts.sort()
    p95 = dts[int(len(dts) * 0.95)]
    print("===== 纯编排开销（无 LLM，30 次）=====")
    print(f"均值: {statistics.mean(dts):.3f}s | P95: {p95:.3f}s | 最大: {dts[-1]:.3f}s")


if __name__ == "__main__":
    main()
