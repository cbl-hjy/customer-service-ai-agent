#!/usr/bin/env python3
"""
轻量压测（W3）：批量工单串行处理，统计耗时 / 错误率 / 升级率。

用法：python bench.py [N]
N：压测条数（默认 10，复用工单集 8 条 + 2 变体；真实 API，单条约 10-13s）

注意：真实 LLM 调用，会消耗 API 额度。结果用于 MVP 基线记录（非重负载压测）。
"""

import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_customer_service import make_graph

# 工单集 8 条 + 2 变体
BASE_TICKETS = [
    "你们最新款手机 X3 Lite 多少钱？和 X2 Max 比哪个性价比高？",
    "我的手机升级系统后一直闪退，重启也没用",
    "昨天买的耳机想退掉，7 天无理由怎么申请？",
    "物流太慢了，等了一周还没到，我要投诉你们！",
    "你们客服几点上班？电话多少？",
    "帮我查一下订单 20260812 的物流进度",
    "帮我写一首关于夏天的诗",
    "退款 10 天没到账，中间催了 3 次没人理，我要找你们经理",
]
VARIANT_TICKETS = [
    "平板 Tab Pro 支持触控笔吗？多少钱？",
    "发票抬头可以修改吗？怎么改？",
]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tickets = (BASE_TICKETS + VARIANT_TICKETS)[:n]

    app = make_graph()
    results: list[dict] = []
    errors = 0

    print(f"压测开始：{len(tickets)} 条工单（串行，真实 API）")
    for i, q in enumerate(tickets):
        t0 = time.time()
        try:
            r = app.invoke(
                {"customer_query": q},
                {"configurable": {"thread_id": f"bench-{i}"}},
            )
            dt = time.time() - t0
            results.append({
                "q": q[:18],
                "dt": dt,
                "type": r["query_type"],
                "escalated": r["escalated"],
            })
            print(f"[{i+1}/{len(tickets)}] {dt:5.1f}s {r['query_type']:18s} escal={r['escalated']} | {q[:18]}")
        except Exception as e:  # noqa: BLE001 - 压测统计所有异常
            errors += 1
            print(f"[{i+1}/{len(tickets)}] ERROR: {q[:20]} -> {e}")

    # 统计
    dts = [x["dt"] for x in results]
    n_escalated = sum(1 for x in results if x["escalated"])
    n_guardrail = sum(1 for x in results if x["type"] == "out_of_scope")
    print("\n===== 压测汇总 =====")
    print(f"总条数: {len(tickets)} | 成功: {len(results)} | 错误: {errors} | 错误率: {errors / len(tickets):.1%}")
    if dts:
        print(f"耗时: 总 {sum(dts):.1f}s | 均值 {statistics.mean(dts):.2f}s | 中位 {statistics.median(dts):.2f}s | 最大 {max(dts):.2f}s")
    print(f"升级率: {n_escalated}/{len(tickets)} | 护栏拒绝: {n_guardrail}/{len(tickets)}")
    print(f"压测时间: {time.strftime('%Y-%m-%d %H:%M:%S')} | 模型: qwen3.7-plus")


if __name__ == "__main__":
    main()
