#!/usr/bin/env python3
"""实验 C：未命中归因分析（第三波 RAG 优化，2026-09-16）。

问题：组件金标 217 条中 rerank R@1 未命中约一半——需区分三类根因，
决定下一波是"扩 KB"还是"调检索"还是"修金标"：
  1. 召回失败（expected 从未进入 RRF 池 top-20）：BM25+dense 双路都没召回
     → 若 KB 有对应条目属检索改进项；若 KB 真缺属扩容项
  2. 精排失败（expected 进了池但被 rerank 挤出 top5）：召回对了排序错了
     → reranker 质量或池竞争问题；RRF 本来排对而 rerank 挤掉的单独标记
  3. 金标存疑（expected 部分在 top5 但不在 top1）：属排序微差非全 miss，单独统计

方法：纯只读分析 details.json，按上述规则归类，输出三分类计数 + 明细清单
（供 KB 扩容清单与检索改进决策）。

用法：python eval/kb_v2/analyze_misses.py
"""
import json
import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
DETAILS = os.path.join(HERE, "eval_retrieval_rerank_details.json")


def main():
    records = json.load(open(DETAILS, encoding="utf-8"))
    n = len(records)

    miss_recall, miss_rerank, rerank_harm, partial_hit = [], [], [], 0
    for rec in records:
        expected = set(rec["expected"])
        rerank_top5, rrf_top5, pool = rec["rerank_top5"], rec["rrf_top5"], rec["rrf_pool"]
        top1_ok = bool(rerank_top5) and rerank_top5[0] in expected
        if top1_ok:
            continue
        if any(t in expected for t in rerank_top5):
            partial_hit += 1  # 排序微差：expected 在 top5 内但非 top1
            continue
        if any(t in expected for t in pool):
            # 进了池但精排挤出 top5
            if rrf_top5 and rrf_top5[0] in expected:
                rerank_harm.append(rec)  # RRF 本来 top1 正确，rerank 反而挤掉（跳过策略目标区）
            else:
                miss_rerank.append(rec)
        else:
            miss_recall.append(rec)  # 双路召回都没进池

    print(f"归因基座：{n} 条金标 | rerank R@1 未命中 {len(miss_recall) + len(miss_rerank) + len(rerank_harm)}"
          f" + top5 内非 top1 {partial_hit}\n")
    print(f"{'类别':<6} {'条数':>4} {'占比':>7}  含义")
    print(f"{'召回失败':<6} {len(miss_recall):>4} {len(miss_recall) / n:>7.1%}  双路未召回进池（扩KB/召回改进）")
    print(f"{'精排失败':<6} {len(miss_rerank):>4} {len(miss_rerank) / n:>7.1%}  池内有但挤出 top5（排序问题）")
    print(f"{'精排反噬':<6} {len(rerank_harm):>4} {len(rerank_harm) / n:>7.1%}  RRF top1 本对被 rerank 挤掉")
    print(f"{'排序微差':<6} {partial_hit:>4} {partial_hit / n:>7.1%}  expected 在 top5 非 top1\n")

    def dump(name, recs, limit=15):
        if not recs:
            return
        print(f"--- {name}（前 {min(limit, len(recs))} 条）---")
        for rec in recs[:limit]:
            exp = "/".join(rec["expected"][:2])
            pool_hit = next((t for t in rec["rrf_pool"] if t in set(rec["expected"])), None)
            rrf_top1 = rec["rrf_top5"][0] if rec["rrf_top5"] else "-"
            rerank_top1 = rec["rerank_top5"][0] if rec["rerank_top5"] else "-"
            print(f"  [{rec['domain']}/{rec['diff']}] {rec['query']}")
            print(f"    期望: {exp} | 池内命中: {pool_hit} | rrf_top1: {rrf_top1} | rerank_top1: {rerank_top1}")
        print()

    dump("召回失败", miss_recall)
    dump("精排失败", miss_rerank)
    dump("精排反噬", rerank_harm)

    # 召回失败明细落盘（KB 扩容候选清单）
    if miss_recall:
        out_path = os.path.join(HERE, "kb_gap_candidates.json")
        payload = [
            {"query": r["query"], "domain": r["domain"], "difficulty": r["diff"],
             "expected": r["expected"]}
            for r in miss_recall
        ]
        json.dump(payload, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"召回失败清单已落盘（KB 扩容候选）：{os.path.basename(out_path)}（{len(payload)} 条）")


if __name__ == "__main__":
    main()
