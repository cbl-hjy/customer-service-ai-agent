#!/usr/bin/env python3
"""OOD 升级触发评估（banking77 作为域外样本集）。

banking77 的英文银行意图相对电商客服系统 = 域外合理客服问题。
验证 fail-safe：域外问题应升级（不硬答），而非被一线 agent 编造回答。

用法：python eval_ood.py [样本数] [并发]
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

from multi_agent_customer_service import make_graph

_app = make_graph()  # 模块级全局实例（诊断实测：单例化 getter 在 ThreadPool 下有偶发问题）


def _run(i: int, text: str) -> dict:
    t0 = time.time()
    try:
        r = _app.invoke(
            {"customer_query": text},
            {"configurable": {"thread_id": f"ood-{i}"}},
        )
        return {"ok": True, "dt": time.time() - t0, "escalated": r["escalated"], "type": r["query_type"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "dt": time.time() - t0, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    rows = [json.loads(l) for l in open("data/banking77_test.jsonl", encoding="utf-8")][:n]
    print(f"评估：banking77 前 {n} 条（域外样本）| 并发 {concurrency}")

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_run, i, r["text"]): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            results.append(fut.result())

    oks = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    n_escalated = sum(1 for r in oks if r["escalated"])
    n_normal = sum(1 for r in oks if not r["escalated"])
    dts = [r["dt"] for r in oks]

    print("\n===== OOD 升级触发评估 =====")
    print(f"样本: {len(rows)} | 成功: {len(oks)} | 错误: {len(errs)}")
    if errs:
        from collections import Counter

        for err, cnt in Counter(e["error"] for e in errs).most_common(5):
            print(f"  错误[{cnt}]: {err}")
    print(f"升级率（正确 fail-safe 行为）: {n_escalated}/{len(oks)} = {n_escalated / len(oks):.1%}" if oks else "无成功样本")
    print(f"未升级（风险：可能硬答域外问题）: {n_normal}/{len(oks)} = {n_normal / len(oks):.1%}" if oks else "")
    if dts:
        print(f"耗时: 总 {sum(dts):.0f}s | 均值 {sum(dts)/len(dts):.1f}s")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
