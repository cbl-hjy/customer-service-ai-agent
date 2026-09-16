#!/usr/bin/env python3
"""实验 A：reranker 置信度跳过仿真（第三波 RAG 优化，2026-09-16）。

问题：reranker 精排 20 候选 0.6-2s/次（CPU），占端到端延迟大头；但部分查询
RRF 融合信号已足够强（词面与语义双路一致），精排只是确认甚至反噬
（details 中有 rerank 把 RRF 正确 top1 挤掉的实例）。

方法：纯离线仿真——只用 eval_retrieval_rerank_details.json 已有数据，
模拟"强信号查询跳过精排（用 RRF 序），其余走精排"。跳过策略只用
运行时可得的信号（bm25/rrf 排序一致性），不使用 expected（防答案泄漏）。

决策门（方案定标）：跳过率 ≥30% 且跳过子集 R@1 损失 <1pp 才建议线上实施。

用法：python eval/kb_v2/simulate_rerank_skip.py
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


def eval_ranks(titles, expected):
    """与 eval_retrieval_rerank.py 同口径：R@1 与 MRR。"""
    r1 = 1.0 if titles and titles[0] in expected else 0.0
    mrr = 0.0
    for i, t in enumerate(titles):
        if t in expected:
            mrr = 1.0 / (i + 1)
            break
    return r1, mrr


def policy_p1(rec):
    """P1 宽松：BM25 top1 与 RRF top1 一致（词面与融合双路同指）。"""
    return rec["bm25_top5"] and rec["rrf_top5"] and rec["bm25_top5"][0] == rec["rrf_top5"][0]


def policy_p2(rec):
    """P2 严格：top2 顺位完全一致（双路排序强收敛）。"""
    return (rec["bm25_top5"][:2] == rec["rrf_top5"][:2])


def simulate(records, policy_fn):
    """按策略仿真：命中策略的查询用 rrf_top5，其余用 rerank_top5。"""
    skipped, hybrid_r1, hybrid_mrr, rerank_r1, rerank_mrr = 0, 0.0, 0.0, 0.0, 0.0
    skip_r1_rrf, skip_r1_rerank = 0.0, 0.0  # 跳过子集上两路各自的 R@1
    for rec in records:
        expected = set(rec["expected"])
        r_rr, m_rr = eval_ranks(rec["rerank_top5"], expected)
        r_rf, m_rf = eval_ranks(rec["rrf_top5"], expected)
        rerank_r1 += r_rr
        rerank_mrr += m_rr
        if policy_fn(rec):
            skipped += 1
            hybrid_r1 += r_rf
            hybrid_mrr += m_rf
            skip_r1_rrf += r_rf
            skip_r1_rerank += r_rr
        else:
            hybrid_r1 += r_rr
            hybrid_mrr += m_rr
    n = len(records)
    return {
        "skip_rate": skipped / n,
        "hybrid_r@1": hybrid_r1 / n,
        "hybrid_mrr": hybrid_mrr / n,
        "rerank_r@1": rerank_r1 / n,
        "rerank_mrr": rerank_mrr / n,
        "skip_subset_r@1_rrf": skip_r1_rrf / skipped if skipped else None,
        "skip_subset_r@1_rerank": skip_r1_rerank / skipped if skipped else None,
        "skipped": skipped,
    }


def main():
    records = json.load(open(DETAILS, encoding="utf-8"))
    n = len(records)
    print(f"仿真基座：{n} 条金标（当前 KB 组件评估 details）\n")

    # 参照系：三路全量
    full_rerank_r1 = sum(1 for r in records if r["rerank_top5"] and r["rerank_top5"][0] in set(r["expected"])) / n
    full_rrf_r1 = sum(1 for r in records if r["rrf_top5"] and r["rrf_top5"][0] in set(r["expected"])) / n
    print(f"参照系：全量 rerank R@1={full_rerank_r1:.3f} | 全量 RRF R@1={full_rrf_r1:.3f}\n")

    for name, fn in [("P1 top1一致", policy_p1), ("P2 top2顺位一致", policy_p2)]:
        s = simulate(records, fn)
        print(f"策略 {name}: 跳过率 {s['skip_rate']:.1%}（{s['skipped']}/{n}）")
        print(f"  混合 R@1={s['hybrid_r@1']:.3f}（vs 全量 rerank {s['rerank_r@1']:.3f}，"
              f"Δ={s['hybrid_r@1'] - s['rerank_r@1']:+.3f}） | MRR={s['hybrid_mrr']:.3f}"
              f"（Δ={s['hybrid_mrr'] - s['rerank_mrr']:+.3f}）")
        if s["skipped"]:
            loss = s["skip_subset_r@1_rerank"] - s["skip_subset_r@1_rrf"]
            print(f"  跳过子集：RRF R@1={s['skip_subset_r@1_rrf']:.3f} vs rerank R@1="
                  f"{s['skip_subset_r@1_rerank']:.3f}（跳过造成 {loss:+.3f}）")
        verdict = "✅ 过决策门" if (s["skip_rate"] >= 0.30 and s["skipped"]
                                    and (s["skip_subset_r@1_rerank"] - s["skip_subset_r@1_rrf"]) < 0.01
                                    and s["hybrid_r@1"] >= s["rerank_r@1"]) else "❌ 未过决策门"
        print(f"  决策门（跳过率≥30% 且子集损失<1pp 且整体不降）：{verdict}\n")


if __name__ == "__main__":
    main()
