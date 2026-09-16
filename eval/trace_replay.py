#!/usr/bin/env python3
"""T2 生产 trace 回流重放（2026-09-16）：真实分布驱动的评估闭环入口。

对标背景：Nubank 评估驱动闭环（"评估管线质量直接决定迭代速度"）+ EdsDev reality
gap（生产准确率仅为评估集 40-50%——病根是评估集与真实用户不同源）。本项目无真实
流量，回流池 = web demo / 压测 / 人工测试累计的生产形态对话（trace.db，排除评估轮）。

管线（五层）：
  1. 生产池聚合：trace.db 过滤（eval-/v12-/v14-/bench- 前缀）→ unique query
     （历史决策 label 分布 + 轮次；demo 高度重复，unique 才是分布单位）
  2. KB 覆盖检查：按历史主导 label 的域跑 kb_retrieve（确定性，零成本）——
     miss = KB 覆盖缺口候选（reality gap 的确定性代理）
  3. 重放：每 unique query 用当前系统重跑一次（app.invoke，新 thread）——
     历史回复未采集（T2 之前无 response 列），重放语义 = 当前行为快照
  4. 审计：确定性检查（内部泄漏/系统错误/空回复）+ judge 事实一致性
     （当前 KB 基座；tool 轮拼订单快照）
  5. 决策一致性 + 候选沉淀：历史主导 label vs 重放 label（行为漂移检测——
     RRF 调参/fp16/T1 之后决策变了多少）；judge fail / KB miss / 漂移 →
     eval/golden_candidates.json（append，人工审核入金标的入口）

隔离纪律（与 eval_multi_turn 同源）：重放写 eval 侧 checkpoints/orders/trace，
生产库只读（聚合在隔离变量设置之前完成）。

用法（包根目录）：
  python eval/trace_replay.py                # 全量 unique 重放（judge 全开）
  python eval/trace_replay.py --limit 10     # 只跑前 10 条（管线验证）
  python eval/trace_replay.py --no-judge     # 关 judge（只确定性检查+分布）
  python eval/trace_replay.py --multi-turn   # 多轮 thread 聚合回流（checkpoints.db 源）

多轮模式（w8 挂账，2026-09-16）：源 = 生产 checkpoints.db 的 persisted_dialogue
（权威多轮记录；历史 trace thread_id 缺 session_id 已废不可聚合，web 层透传修复
只惠及增量）。过滤 = 排除前缀 thread + 垃圾轮 + 去重后 <2 不同 query 的重复型
thread（压测同句重放形态，语义非多轮）。重放 = 同一 eval 侧 thread 顺序 invoke
（带上下文延续），逐轮审计；指代轮 OOD 误判 → context_lost 候选。
"""
import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, PACKAGE_ROOT)
sys.path.insert(0, EVAL_DIR)

# 生产池排除前缀：评估轮（eval-）、压测轮（v12-/v14-/bench-）——非生产形态
_EXCLUDE_PREFIXES = ("eval-", "v12-", "v14-", "bench-")

_LABEL2DOMAIN = {
    "complaint": "complaint", "billing": "billing",
    "technical_support": "tech", "product_info": "product",
    "general_inquiry": "general",
}
_ORDER_TOOL_NAMES = ("query_order", "query_logistics", "update_order_address")
_ORDER_ID_RE = re.compile(r"XH\d{12}")


def load_production_pool(db_path: str) -> list:
    """聚合生产池 unique query（只读生产 trace.db，在隔离变量设置前调用）。"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    like = " OR ".join(f"thread_id LIKE '{p}%'" for p in _EXCLUDE_PREFIXES)
    rows = conn.execute(
        f"SELECT user_query, ts, decision FROM trace_runs WHERE NOT ({like})"
    ).fetchall()
    pool = {}
    for r in rows:
        q = r["user_query"]
        if len(q.strip()) < 2:
            continue  # 测试垃圾 query（单字符/空串）——非生产形态
        try:
            dec = json.loads(r["decision"] or "{}")
        except Exception:
            dec = {}
        slot = pool.setdefault(q, {"query": q, "n_turns": 0, "labels": Counter(),
                                   "escalated": 0, "latest_ts": ""})
        slot["n_turns"] += 1
        if dec.get("query_type"):
            slot["labels"][dec["query_type"]] += 1
        if dec.get("escalated"):
            slot["escalated"] += 1
        if r["ts"] > slot["latest_ts"]:
            slot["latest_ts"] = r["ts"]
    conn.close()
    out = list(pool.values())
    out.sort(key=lambda s: -s["n_turns"])  # 高频在前
    return out


def load_multi_turn_threads(ckpt_db_path: str) -> list:
    """聚合生产 checkpoints.db 的多轮 thread（只读纪律：复制副本 + SqliteSaver API 读副本）。

    必须在设置 CHECKPOINT_DB_PATH 隔离变量之前调用（生产库只读）。
    过滤：排除前缀 thread（评估/压测/replay 形态）→ 垃圾轮（<2 字符）→
    去重后 <2 不同 query 的重复型 thread（压测同句重放，语义非多轮）。
    """
    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.sqlite import SqliteSaver

    dst = os.path.join(EVAL_DIR, "eval_data", f"_mt_src_copy_{os.getpid()}.db")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(ckpt_db_path, dst)
    try:
        conn = sqlite3.connect(dst)
        threads = [r[0] for r in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")]
        conn.close()
        prod = [t for t in threads if not t.startswith(_EXCLUDE_PREFIXES + ("replay-", "smoke-"))]

        cp_cm = SqliteSaver.from_conn_string(dst)
        cp = cp_cm.__enter__()
        out = []
        try:
            for tid in prod:
                try:
                    tup = cp.get_tuple({"configurable": {"thread_id": tid}})
                except Exception:
                    continue
                if tup is None:
                    continue
                pd = (tup.checkpoint.get("channel_values") or {}).get("persisted_dialogue") or []
                turns = [str(m["content"]).strip() for m in pd
                         if isinstance(m, dict) and m.get("is_user")
                         and len(str(m.get("content", "")).strip()) >= 2]
                # 重复型 thread（去重后仅 1 个 query）：压测同句重放形态，非多轮语义
                if len(set(turns)) < 2:
                    continue
                out.append({"thread_id": tid, "turns": turns,
                            "n_msgs": len(pd)})
        finally:
            cp_cm.__exit__(None, None, None)
        out.sort(key=lambda s: -len(s["turns"]))
        return out
    finally:
        if os.path.exists(dst):
            os.remove(dst)


def main():
    parser = argparse.ArgumentParser(description="T2 生产 trace 回流重放")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条 unique/thread（0=全量）")
    parser.add_argument("--no-judge", action="store_true", help="关 judge（只确定性检查+分布）")
    parser.add_argument("--multi-turn", action="store_true",
                        help="多轮 thread 聚合回流（源=生产 checkpoints.db）")
    parser.add_argument("--db", default=os.path.join(PACKAGE_ROOT, "data", "trace.db"),
                        help="生产 trace.db 路径（默认 data/trace.db）")
    parser.add_argument("--ckpt-db", default=os.path.join(PACKAGE_ROOT, "data", "checkpoints.db"),
                        help="生产 checkpoints.db 路径（--multi-turn 用）")
    args = parser.parse_args()

    # ---- 1. 生产池聚合（先读生产库，再进隔离环境） ----
    hist_pool = {s["query"]: s for s in load_production_pool(args.db)}
    if args.multi_turn:
        pool = load_multi_turn_threads(args.ckpt_db)
        if args.limit:
            pool = pool[:args.limit]
        print(f"📦 多轮回流池：{len(pool)} threads（真多轮：去重后 ≥2 不同 query）")
    else:
        pool = list(hist_pool.values())
        if args.limit:
            pool = pool[:args.limit]
        print(f"📦 生产回流池：{len(pool)} unique queries")

    # ---- 2. 进入评估隔离环境（重放写 eval 侧，生产库不再触碰） ----
    os.environ["CHECKPOINT_DB_PATH"] = os.path.join(EVAL_DIR, "eval_data", "checkpoints_eval.db")
    os.environ["ORDERS_DB_PATH"] = os.path.join(EVAL_DIR, "eval_data", "orders_eval.db")
    os.environ["TRACE_DB_PATH"] = os.path.join(EVAL_DIR, "eval_data", "trace_eval.db")

    import eval_multi_turn as _emt  # noqa: E402  复用：隔离初始化 + 确定性检查 + judge
    from multi_agent_customer_service import make_graph, reset_token_usage  # noqa: E402
    from kb_retriever import retrieve as _kb_retrieve  # noqa: E402

    app = make_graph()
    reset_token_usage()

    judge_on = not args.no_judge

    def _audit_one(rec, q, state, hist_slot):
        """单轮审计（单轮/多轮共用）：KB 覆盖 + 确定性检查 + 决策一致性 + judge。"""
        response = str(state.get("response") or "")
        tools_used = list(state.get("tools_used") or [])
        label = state.get("query_type", "")
        rec.update({
            "replay_label": label,
            "replay_confidence": round(float(state.get("query_confidence", 0.0)), 3),
            "replay_escalated": bool(state.get("escalated")),
            "replay_tools": [t for t in tools_used if t in _ORDER_TOOL_NAMES],
            "response_full": response,
        })
        # KB 覆盖（按重放 label 的域，确定性）
        dom = _LABEL2DOMAIN.get(label)
        rec["kb_hit"] = None
        if dom:
            try:
                rec["kb_hit"] = bool(_kb_retrieve(dom, q))
            except Exception:
                rec["kb_hit"] = None
        # 确定性审计（复用 eval_multi_turn 的检查器）
        rec["checks"] = {
            "no_internal_leak": _emt._no_internal_leak(response),
            "no_system_error": _emt._no_system_error(response),
            "non_empty": bool(response.strip()),
        }
        # 决策一致性：与 trace 单 query 池历史主导交叉
        rec["decision_drift"] = None
        if hist_slot and hist_slot["labels"]:
            dominant = hist_slot["labels"].most_common(1)[0][0]
            rec["decision_drift"] = (label != dominant)
        # judge（当前 KB 基座；tool 轮拼订单快照）
        rec["judge"] = None
        if judge_on and label != "out_of_scope" and not state.get("escalated"):
            domain = _LABEL2DOMAIN.get(label, "general")
            cited = _emt._parse_citations(response)
            kb_ctx = _emt._kb_context_for_judge(domain, q, cited)
            if any(t in _ORDER_TOOL_NAMES for t in tools_used):
                from tools.orders import query_order as _qo
                oids = _ORDER_ID_RE.findall(q) or _ORDER_ID_RE.findall(response)
                snap = "\n\n".join(_qo(o) for o in set(oids))
                if snap:
                    kb_ctx = f"{kb_ctx}\n\n【订单系统数据】\n{snap}".strip()
            if kb_ctx.strip():
                j_pass, j_detail = _emt._judge_faithfulness(response, kb_ctx, _emt.get_llm())
                rec["judge"] = {"pass": j_pass, "detail": j_detail[:200]}
        return rec

    if not args.multi_turn:
        # ---- 3. 单轮模式：KB 覆盖预检（保持原口径：按历史主导 label 域） ----
        print("🔍 KB 覆盖检查...")
        for slot in pool:
            dom = _LABEL2DOMAIN.get(slot["labels"].most_common(1)[0][0] if slot["labels"] else "")
            slot["kb_hit"] = None  # OOD / 无主导 label：不检查
            if dom:
                try:
                    slot["kb_hit"] = bool(_kb_retrieve(dom, slot["query"]))
                except Exception:
                    slot["kb_hit"] = None

    # ---- 4. 重放 + 审计 ----
    print(f"▶️ 重放（judge={'on' if judge_on else 'off'}，串行）...")
    t0 = time.time()
    results = []

    if args.multi_turn:
        from graph.checkpointer import get_checkpointer  # noqa: E402
        for i, th in enumerate(pool, 1):
            rtid = f"replay-mt-{abs(hash(th['thread_id'])) & 0xffffff:x}"
            # 清 eval 侧同名残留 thread（重跑幂等：否则旧 persisted_dialogue 叠加污染）
            try:
                get_checkpointer().delete_thread(rtid)
            except Exception:
                pass
            for turn_idx, q in enumerate(th["turns"], 1):
                rec = {"mode": "multi_turn", "thread_id": th["thread_id"],
                       "replay_thread": rtid, "turn_idx": turn_idx, "turns_total": len(th["turns"]),
                       "query": q, "hist_labels": dict(hist_pool[q]["labels"]) if q in hist_pool else {}}
                try:
                    # session_id 注入（与 web 层修复同构）：多轮上下文延续 + trace 聚合键
                    state = app.invoke({"customer_query": q, "session_id": rtid},
                                       {"configurable": {"thread_id": rtid}})
                    _audit_one(rec, q, state, hist_pool.get(q))
                except Exception as e:  # noqa: BLE001
                    rec.update({"replay_error": str(e)[:200]})
                    results.append(rec)
                    print(f"  [{i}/{len(pool)}] T{turn_idx} ERROR {q[:40]}: {e}")
                    continue
                # 多轮特有：上下文丢失信号——非首轮被 OOD 拒答，但该 query 历史主导非 OOD
                rec["context_lost"] = None
                if turn_idx > 1 and rec.get("replay_label") == "out_of_scope":
                    hist = rec.get("hist_labels") or {}
                    if hist and max(hist, key=hist.get) != "out_of_scope":
                        rec["context_lost"] = True
                results.append(rec)
                flag = ""
                if rec.get("decision_drift"):
                    flag += " ⚠️drift"
                if rec.get("kb_hit") is False:
                    flag += " ⚠️kb-miss"
                if rec.get("judge") and not rec["judge"]["pass"]:
                    flag += " ❌judge"
                if rec.get("context_lost"):
                    flag += " ⚠️ctx-lost"
                print(f"  [{i}/{len(pool)}] T{turn_idx}/{len(th['turns'])} {q[:40]}{flag}")
    else:
        for i, slot in enumerate(pool, 1):
            q = slot["query"]
            rec = {"query": q, "n_turns": slot["n_turns"],
                   "hist_labels": dict(slot["labels"]), "hist_escalation_rate": round(slot["escalated"] / slot["n_turns"], 2),
                   "latest_ts": slot["latest_ts"], "kb_hit": slot["kb_hit"]}
            try:
                state = app.invoke({"customer_query": q},
                                   {"configurable": {"thread_id": f"replay-{abs(hash(q)) & 0xffffff:x}"}})
            except Exception as e:  # noqa: BLE001
                rec.update({"replay_error": str(e)[:200]})
                results.append(rec)
                print(f"  [{i}/{len(pool)}] ERROR {q[:40]}: {e}")
                continue
            _audit_one(rec, q, state, slot)
            results.append(rec)
            flag = ""
            if rec.get("decision_drift"):
                flag += " ⚠️drift"
            if rec.get("kb_hit") is False:
                flag += " ⚠️kb-miss"
            if rec.get("judge") and not rec["judge"]["pass"]:
                flag += " ❌judge"
            print(f"  [{i}/{len(pool)}] {q[:44]}{flag}")

    elapsed = time.time() - t0

    # ---- 5. 汇总 + 金标候选沉淀 ----
    def _is_candidate(rec):
        reasons = []
        if rec.get("kb_hit") is False:
            reasons.append("kb_miss: 当前 KB 检索未覆盖该真实 query（覆盖缺口）")
        if rec.get("decision_drift"):
            reasons.append("decision_drift: 重放决策与历史主导决策不一致（行为漂移）")
        if rec.get("judge") and not rec["judge"]["pass"]:
            reasons.append(f"judge_fail: {rec['judge']['detail']}")
        if not all(rec.get("checks", {}).values()):
            reasons.append(f"check_fail: {[k for k, v in rec.get('checks', {}).items() if not v]}")
        if rec.get("context_lost"):
            reasons.append("context_lost: 多轮语境下追问被 OOD 拒答（上下文未延续）")
        return reasons

    candidates = []
    for rec in results:
        reasons = _is_candidate(rec)
        if reasons:
            cand = {
                "query": rec["query"],
                "mode": rec.get("mode", "single"),
                "reasons": reasons,
                "hist_labels": rec.get("hist_labels"),
                "replay_label": rec.get("replay_label"),
                "kb_hit": rec.get("kb_hit"),
                "response_excerpt": str(rec.get("response_full", ""))[:200],
                "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if rec.get("mode") == "multi_turn":
                cand["thread_id"] = rec.get("thread_id")
                cand["turn_idx"] = rec.get("turn_idx")
            candidates.append(cand)

    # 分布对比（生产 unique vs 生产轮次加权 + 审计汇总）
    label_uniq = Counter(r.get("replay_label", "?") for r in results if r.get("replay_label"))
    summary = {
        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "multi_turn" if args.multi_turn else "single",
        "pool_unique": len(pool),
        "pool_turns_total": sum(len(s["turns"]) for s in pool) if args.multi_turn
                           else sum(s["n_turns"] for s in pool),
        "replay_elapsed_s": round(elapsed, 1),
        "judge_enabled": judge_on,
        "replay_label_distribution_unique": dict(label_uniq),
        "kb_miss_count": sum(1 for r in results if r.get("kb_hit") is False),
        "decision_drift_count": sum(1 for r in results if r.get("decision_drift")),
        "judge_fail_count": sum(1 for r in results if r.get("judge") and not r["judge"]["pass"]),
        "check_fail_count": sum(1 for r in results if not all(r.get("checks", {}).values())),
        "candidate_count": len(candidates),
    }
    if args.multi_turn:
        summary["context_lost_count"] = sum(1 for r in results if r.get("context_lost"))

    # 报告落盘
    os.makedirs(os.path.join(EVAL_DIR, "reports"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"trace_replay_multi_{stamp}.json" if args.multi_turn else f"trace_replay_{stamp}.json"
    report_path = os.path.join(EVAL_DIR, "reports", report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=1)

    # 金标候选沉淀（append，按 (query, mode) 去重）
    cand_path = os.path.join(EVAL_DIR, "golden_candidates.json")
    new_keys = {(x["query"], x.get("mode", "single")) for x in candidates}
    existing = []
    if os.path.exists(cand_path):
        try:
            existing = [c for c in json.load(open(cand_path, encoding="utf-8"))
                        if (c.get("query"), c.get("mode", "single")) not in new_keys]
        except Exception:
            existing = []
    with open(cand_path, "w", encoding="utf-8") as f:
        json.dump(existing + candidates, f, ensure_ascii=False, indent=1)

    # 控制台摘要
    print("\n" + "=" * 60)
    unit = "threads" if args.multi_turn else "unique"
    print(f"📦 回流池: {summary['pool_unique']} {unit} / {summary['pool_turns_total']} 轮")
    print(f"🏷 重放 label 分布(轮级): {summary['replay_label_distribution_unique']}")
    print(f"🔍 KB 未覆盖: {summary['kb_miss_count']} | ⚠️ 决策漂移: {summary['decision_drift_count']} "
          f"| ❌ judge fail: {summary['judge_fail_count']} | 检查 fail: {summary['check_fail_count']}"
          + (f" | ⚠️ 上下文丢失: {summary['context_lost_count']}" if args.multi_turn else ""))
    print(f"⏱ 重放耗时 {summary['replay_elapsed_s']}s")
    print(f"📄 报告: {report_path}")
    print(f"🥇 金标候选: {summary['candidate_count']} 条 → {cand_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
