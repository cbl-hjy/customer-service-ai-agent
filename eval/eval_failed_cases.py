#!/usr/bin/env python3
"""
专项复现排查：对已知失败 case 重复运行 N 次，采集分层数据判断【偶发性】。
目的：不到草率定方案——先用数据区分"偶发抖动"vs"稳定缺陷"，再定位到分类/检索/生成/路由哪一层。

分层字段：
  分类层: query_type / query_confidence
  决策层: escalated / escalation_reason
  生成层: response（截断）
  检索层: 由 response 是否含"知识库暂未收录/转人工"间接推断（agent 无命中触发 _no_answer 升级）

用法: python eval/eval_failed_cases.py <case_id> [--runs N] [--turn T]
      默认 runs=5, turn=0(该 case 全部轮)
"""
import os
import sys
import json
import time
import datetime as _dt

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DB = os.path.join(EVAL_DIR, "eval_data", "checkpoints_eval.db")
os.environ["CHECKPOINT_DB_PATH"] = EVAL_DB
sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)

from multi_agent_customer_service import make_graph  # noqa: E402


def run_case_once(app, case, run_idx):
    """单次全轮运行，返回每轮分层数据。"""
    cid = case["id"]
    tid = f"repro-{cid}-r{run_idx}"
    rows = []
    for i, turn in enumerate(case["turns"]):
        user_text = turn["user"]
        expect = turn.get("expect_decision", "answer")
        try:
            result = app.invoke(
                {"customer_query": user_text},
                {"configurable": {"thread_id": tid}},
            )
            error = None
        except Exception as e:  # noqa: BLE001
            result = {}
            error = str(e)
        if error:
            rows.append({"turn": i + 1, "user": user_text, "expect": expect, "error": error})
            continue
        resp = str(result.get("response") or "")
        # 检索层命中推断：含转人工/未收录 → 检索无命中
        retrieval_miss = ("知识库暂未收录" in resp) or ("转人工" in resp) or ("转接人工" in resp)
        rows.append({
            "turn": i + 1,
            "user": user_text,
            "expect": expect,
            "query_type": result.get("query_type", ""),
            "confidence": round(float(result.get("query_confidence", 0.0)), 3),
            "escalated": bool(result.get("escalated")),
            "escalation_reason": result.get("escalation_reason", ""),
            "current_agent": result.get("current_agent", ""),
            "retrieval_miss_hint": retrieval_miss,
            "resp": resp[:120].replace("\n", " "),
        })
    return rows


def main():
    args = [a for a in sys.argv[1:]]
    runs = 5
    turn_filter = 0
    if "--runs" in args:
        runs = int(args[args.index("--runs") + 1])
    if "--turn" in args:
        turn_filter = int(args[args.index("--turn") + 1])
    case_ids = args[0].split(",") if args else ["mt-002", "mt-003", "mt-004"]

    golden_path = os.environ.get("GOLDEN_PATH") or os.path.join(EVAL_DIR, "golden_multi_turn.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    by_id = {c["id"]: c for c in golden["cases"]}

    if os.path.exists(EVAL_DB):
        os.remove(EVAL_DB)

    app = make_graph()

    for cid in case_ids:
        case = by_id.get(cid)
        if not case:
            print(f"⚠ 未找到 case {cid}")
            continue
        print(f"\n{'='*70}\n▶ case {cid} [{case['persona']}]  runs={runs}" + (f"  turn_only={turn_filter}" if turn_filter else ""))
        print(f"{'='*70}")
        per_turn = {}
        for r in range(1, runs + 1):
            rows = run_case_once(app, case, r)
            for row in rows:
                per_turn.setdefault(row["turn"], []).append(row)
        # 汇总
        for t, rows in sorted(per_turn.items()):
            if turn_filter and t != turn_filter:
                continue
            print(f"\n  --- 轮{t} 「{rows[0]['user']}」 expect={rows[0]['expect']} 共{len(rows)}次 ---")
            for row in rows:
                if "error" in row:
                    print(f"    [r error] {row['error']}")
                    continue
                flag = "✓" if row["escalated"] == (row["expect"] == "escalate") else "✗"
                print(f"    {flag} type={row['query_type']:<14} conf={row['confidence']:.2f} "
                      f"esc={row['escalated']} retr_miss={row['retrieval_miss_hint']} agent={row['current_agent']}")
                print(f"        {row['resp']}")
            # 统计该轮决策、分类、检索失配的分布
            if "error" not in rows[0]:
                types = [r["query_type"] for r in rows]
                escs = [r["escalated"] for r in rows]
                miss = [r["retrieval_miss_hint"] for r in rows]
                from collections import Counter
                tdist = dict(Counter(types))
                edist = dict(Counter(escs))
                mdist = dict(Counter(miss))
                print(f"    [分布] query_type={tdist}  escalated={edist}  retr_miss={mdist}")
    print("\n专项复现完成。")


if __name__ == "__main__":
    main()