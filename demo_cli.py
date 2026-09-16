#!/usr/bin/env python3
"""
CLI 演示（C6）：电商客服升级系统——输入工单，可视化完整处理过程。

用法：
    python demo_cli.py                      # 交互模式（循环输入）
    python demo_cli.py "你的工单内容"        # 单条模式

展示：分类（label/confidence/complexity）→ 路由决策（一线/升级/护栏）→ 处理结果。
理念（§3.4）：agent 自由处理，越线由 harness 兜底升级人工。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_customer_service import make_graph

app = make_graph()


def show(result) -> None:
    """打印一条工单的完整处理过程。"""
    print("=" * 58)
    print("📥 工单:", result["customer_query"])
    print("🎯 分类: {}  (conf={:.2f}, complexity={})".format(
        result["query_type"], result["query_confidence"], result["query_complexity"]
    ))
    if result["query_type"] == "out_of_scope":
        print("🚦 路由: 护栏拒绝（out_of_scope，不进入业务处理）")
    elif result["escalated"]:
        print("🚦 路由: 升级人工客服")
        print("   ↳ 原因:", result["escalation_reason"])
        print("   ↳ 摘要:", str(result["escalation_summary"])[:130])
    else:
        print("🚦 路由: 一线处理 →", result["current_agent"])
    print("💬 回复:", str(result["response"])[:160])
    print("=" * 58)


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = app.invoke(
            {"customer_query": query},
            {"configurable": {"thread_id": "cli-demo"}},
        )
        show(result)
    else:
        print("智能客服升级系统 CLI 演示（Ctrl+C 退出）")
        while True:
            try:
                query = input("请输入工单: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            result = app.invoke(
                {"customer_query": query},
                {"configurable": {"thread_id": "cli-demo"}},
            )
            show(result)


if __name__ == "__main__":
    main()
