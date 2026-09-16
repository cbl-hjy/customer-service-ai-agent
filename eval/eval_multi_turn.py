#!/usr/bin/env python3
"""
多轮金标评估（独立进程，隔离生产 DB）

目的：逼近真实商业场景——用"真实多轮对话 + 决策/答案双评分"暴露当前测试看不到的真问题。
原则：
  - 打分在环外（out-of-loop）：只读 agent 返回的 state / response，只写评估报告，绝不改 agent。
  - 进程隔离：独立 checkpoints 库（CHECKPOINT_DB_PATH），全新熔断器/信号量/单例，与线上零交集。
  - 黑盒调用：与 web 层同一 invoke 契约 app.invoke({"customer_query":...}, {"configurable":{"thread_id":...}})。

用法（在包根目录）：python eval/eval_multi_turn.py
运行时必须在 .env 配置真实 API 密钥（需联网调用 LLM）。
"""
import os
import sys
import json
import re
import time
import logging
import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 隔离第一步：在 import 任何会实例化 checkpointer 的模块之前，指定独立评估库
# ---------------------------------------------------------------------------
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DB = os.path.join(EVAL_DIR, "eval_data", "checkpoints_eval.db")
PROD_DB = os.path.join(PACKAGE_ROOT, "data", "checkpoints.db")
os.environ["CHECKPOINT_DB_PATH"] = EVAL_DB

sys.path.insert(0, PACKAGE_ROOT)
os.chdir(PACKAGE_ROOT)  # 保证 .env 与 knowledge_base.json 相对路径正确

from multi_agent_customer_service import make_graph, get_token_usage, reset_token_usage, get_llm  # noqa: E402

# ---------------------------------------------------------------------------
# 答案质量检查特征（确定性，不依赖 LLM 裁判）
# ---------------------------------------------------------------------------
# 内部前缀泄漏：final_response_node 会拼 【{agent}'s Response】，客户可见即缺陷
_INTERNAL_LEAK_RE = re.compile(r"【[^】]*Response】")
_INTERNAL_NAMES = [
    "product_agent", "tech_agent", "billing_agent",
    "complaint_agent", "general_agent",
]
# 系统错误兜底 / 无答案兜底 文案（业务 agent 失败或知识库无匹配时出现）
# V8（2026-08-14）：明确故障词匹配，非裸"系统"子串（mt-201 实证：正常回复含"系统"二字
# 曾被误判失败）；补 "技术问题"（product_agent 兜底文案，其余 4 agent 用"系统错误"）
_SYSTEM_ERROR_TOKENS = [
    "Error: Agent", "Error: No response", "未获取到回复",
    "系统错误", "技术问题", "智能客服服务暂时不可用",
]


def _no_internal_leak(text: str) -> bool:
    if _INTERNAL_LEAK_RE.search(text):
        return False
    for name in _INTERNAL_NAMES:
        if name in text:
            return False
    return True


def _no_system_error(text: str) -> bool:
    return not any(tok in text for tok in _SYSTEM_ERROR_TOKENS)


# A6 引用溯源脚注（harness 统一附加，multi_agent_customer_service._CITATION_FOOTER_PREFIX）
_CITATION_RE = re.compile(r"\n\n参考来源[：:]\s*([^\n]+)")


def _parse_citations(response: str) -> list:
    """从回答解析引用脚注中的条目标题列表（"参考来源：A、B" 拆分）。"""
    m = _CITATION_RE.search(response)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    return [t.strip() for t in re.split(r"[、,，]", raw) if t.strip()]


def _score_faithfulness(response: str, retrieval_hit: bool, escalated: bool) -> tuple:
    """A6 引用溯源检查（确定性，不依赖 LLM 裁判）。

    - 升级/拒答轮（escalated=True）或检索未命中轮：不适用 → (True, "not_applicable")
      （tools_used 是跨轮累积的，升级轮可能残留上轮 _processing 信号，需显式排除 escalated；
      升级回复由 escalate_node 生成，harness 不附加引用）
    - 检索命中轮（本轮 agent 应答）：回答必须带非空"参考来源"脚注（引用真实性由 harness
      保证——harness 与 agent 共用同一检索判定路径，eval 不注入改写器保持纯检索口径，
      不重复检索避免改写器注入差异造成假阳性）。
    返回 (pass, detail)。
    """
    if escalated or not retrieval_hit:
        return True, "not_applicable"
    cited = _parse_citations(response)
    if not cited:
        return False, "检索命中但回答未标注引用（A6 引用溯源缺失）"
    return True, f"cited={cited}"


# =============================================================================
# LLM-as-judge 事实一致性检查（2026-08-17，测量增强——补确定性检查的盲区）
# 确定性答案分（must_contain 等）测不到"回答是否忠于 KB 事实"；judge 把
# 【客服回答】与【回答引用的 KB 条目内容】交给裁判 LLM 核对，抓幻觉/编造。
# 设计：
#   - 只对"检索命中且本轮 agent 应答"的轮跑（升级/拒答轮不适用）
#   - KB 核对基座 = A6 引用脚注解析出的条目内容（harness 与 agent 同路径保证真实命中）
#   - 独立指标 judge_rate + 并入答案分（fail=该轮答案不合格），fail 留存完整原文可审计
#   - 通道异常/解析失败 → not_applicable（fail-safe，judge 故障不误伤答案）
#   - 开关：JUDGE_ENABLED（默认开；06-08 成本低，W4 全量可负担）
# =============================================================================
_JUDGE_ENABLED = os.environ.get("JUDGE_ENABLED", "1") == "1"
_JUDGE_MAX_KB_CHARS = 4000

# agent display name → KB domain（judge 核对基座用；与 multi_agent_customer_service 对应）
_AGENT_DISPLAY2DOMAIN = {
    "产品专家": "product",
    "技术支持专家": "tech",
    "账单专家": "billing",
    "投诉处理专家": "complaint",
    "综合客服": "general",
}

_JUDGE_PROMPT = """你是客服质检员。核对【客服回答】中的事实陈述是否与【知识库内容】一致。

判定规则（只判"明确问题"，合理内容一律 pass）：
1. pass：回答中的事实/数字/政策与知识库一致，或由知识库内容合理推导；回答对知识库未覆盖处做了免责说明（"需确认/建议联系客服"）
2. pass：回答中的操作性/流程性补充（如上门取件、运费垫付、联系客服等客服常规话术）——只要不与知识库明确冲突，不算编造
3. pass：不同场景的时效/政策（如"运费报销时效"与"运费险赔付时效"是两件事）不得互相判矛盾
4. fail：回答与知识库**明确矛盾**（同一场景给出冲突的事实/数字/政策）
5. fail：回答**虚构了知识库明确否定**的关键事实（如知识库说"不支持"，回答说"支持"）

注意：知识库未覆盖 ≠ 幻觉。只有"明确矛盾"或"虚构知识库明确否定的事实"才算 fail。

只输出一个 JSON 对象：{"verdict": "pass" 或 "fail", "reason": "一句话理由"}。"""


def _kb_context_for_judge(domain: str, query: str, cited_titles: list) -> str:
    """judge 核对基座 = agent 实际注入的域检索全文（纯检索口径，与 agent 主路径一致）。

    比"仅引用脚注条目"更完整：agent 回答可能合理使用了检索到的多条目知识
    （引用脚注只标 top-3，发票/快递等条目可能未入引用但已注入 agent 上下文）。
    检索为空（A4 改写重检路径）时回落引用 titles 的条目 content。
    """
    if domain:
        from kb_retriever import retrieve as _kb_retrieve
        try:
            text = _kb_retrieve(domain, query)
            if text and text.strip():
                return text[:_JUDGE_MAX_KB_CHARS]
        except Exception:  # noqa: BLE001  检索失败回落引用基座
            pass
    if cited_titles:
        return _kb_context_from_titles(cited_titles, _JUDGE_MAX_KB_CHARS)
    return ""


def _compound_kb_context_for_judge(subs: list, cited_titles: list) -> str:
    """A5 复合轮 judge 核对基座 = 各子查询跨域检索拼接（与 compound_node 注入一致）。

    普通轮用 current_agent 的单一 domain 检索整句 query；复合轮 agent 实际注入的是
    各子查询按各自 domain 的检索全文（跨域混合），整句 query 在单一域检索会命中无关
    条目（实测 mt-402 返回代收点取件等，缺三包政策/补偿标准），judge 缺依据误判 fail。
    子查询列表从 state.compound_subs 读取（环外只读，与 agent 同拆解源）。检索失败
    回落引用 titles 基座。
    """
    from kb_retriever import retrieve as _kb_retrieve
    parts = []
    for s in subs:
        dom = s.get("domain", "general")
        sub = s.get("sub_query", "")
        try:
            text = _kb_retrieve(dom, sub)
            if text and text.strip():
                parts.append(f"【{dom}域】{text}")
        except Exception:  # noqa: BLE001  单子查询检索失败跳过，其余保留
            continue
    if parts:
        return "\n\n".join(parts)[:_JUDGE_MAX_KB_CHARS]
    if cited_titles:
        return _kb_context_from_titles(cited_titles, _JUDGE_MAX_KB_CHARS)
    return ""


def _kb_context_from_titles(titles: list, max_chars: int = _JUDGE_MAX_KB_CHARS) -> str:
    """按引用标题取 KB 条目 content 拼成核对上下文（title 唯一，跨域搜）。"""
    if not titles:
        return ""
    from kb_retriever import get_retriever
    retriever = get_retriever()
    by_title = {}
    for entries in retriever._data.values():
        for e in entries:
            by_title.setdefault(e.get("title", ""), e)
    parts = []
    for t in titles:
        e = by_title.get(t)
        if e:
            parts.append(f"【{t}】{e.get('content', '')}")
    return "\n".join(parts)[:max_chars]


def _judge_faithfulness(response: str, kb_context: str, llm) -> tuple:
    """LLM 裁判：回答事实是否与 KB 一致（只抓明确矛盾/虚构）。返回 (pass, detail)。"""
    if not _JUDGE_ENABLED or not str(kb_context).strip():
        return True, "not_applicable"
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=_JUDGE_PROMPT),
            HumanMessage(content=f"【客服回答】\n{response}\n\n【知识库内容】\n{kb_context}"),
        ]
        resp = llm.invoke(messages, response_format={"type": "json_object"})
        text = (getattr(resp, "content", "") or "").strip()
        obj = json.loads(text)
        verdict = str(obj.get("verdict", "")).strip().lower()
        reason = str(obj.get("reason", ""))[:150]
        if verdict == "fail":
            return False, f"judge=fail: {reason}"
        return True, f"judge=pass: {reason}"
    except Exception as e:  # noqa: BLE001  judge 通道故障不误伤答案
        logger.warning("LLM judge 调用异常（按不适用处理）: %s", e)
        return True, "not_applicable(judge 异常)"


def _score_answer_checks(response: str, checks: dict) -> dict:
    """对单条回复做确定性答案质量检查。返回 {check: pass/bool}。"""
    results = {}

    must_contain = checks.get("must_contain", [])
    results["must_contain"] = all(kw in response for kw in must_contain)

    # 语义等价同义词：命中任一即通过（适用于 LLM 自然用词变化的场景）
    must_contain_any = checks.get("must_contain_any", [])
    results["must_contain_any"] = any(kw in response for kw in must_contain_any) if must_contain_any else True

    must_not_contain = checks.get("must_not_contain", [])
    results["must_not_contain"] = not any(kw in response for kw in must_not_contain)

    results["no_internal_leak"] = _no_internal_leak(response)
    results["no_system_error"] = _no_system_error(response)
    results["non_empty"] = bool(response.strip())

    # 仅记录、不判分（供报告参考）
    results["_suggested_tone"] = checks.get("suggested_tone", "")
    return results


def _score_decision(state: dict, expected: str) -> tuple:
    """决策评分。返回 (pass: bool, actual: str, detail: str)。

    expected ∈ {answer, escalate, refuse}
    actual 由 escalated 与 query_type 共同决定。
    """
    escalated = bool(state.get("escalated"))
    qtype = state.get("query_type", "")
    if escalated:
        actual = "escalate"
    elif qtype == "out_of_scope":
        actual = "refuse"
    else:
        actual = "answer"
    passed = actual == expected
    return passed, actual, f"expected={expected} actual={actual} label={qtype}"


# 并行评估并发度：case 之间相互独立（独立 thread_id），可并行。
# 上限与全局 API 信号量（MAX_CONCURRENCY=5）一致——API 在途请求由信号量门控，
# 并行不会放大 API 排队；GPU 只被检索 reranker 单例占用，并行不增加显存。
# 串行与并行指标一致（调度不改变 LLM 行为）；仅显著缩短评估墙钟时间。
_EVAL_PARALLEL = max(1, min(int(os.environ.get("EVAL_PARALLEL", "4")), 5))


def _run_case(app, case):
    """执行单个 case 的全部轮（多轮上下文在 case 内顺序保持）。返回 (report, totals_delta, latencies)。

    并发安全：不同 case 用不同 thread_id（eval-{cid}），checkpointer 隔离；
    token 计数有 _TOKEN_LOCK、API 并发有 _API_SEMAPHORE 门控、熔断器有锁。
    """
    cid = case["id"]
    tid = f"eval-{cid}"  # 每 case 独立线程，避免跨 case 串扰
    case_turns = []
    case_ok = True
    ct = {
        "decision_pass": 0, "decision_total": 0,
        "answer_pass": 0, "answer_total": 0,
        "retrieval_attempted": 0, "retrieval_hit": 0,
        "correct_escalate": 0, "false_escalate": 0, "miss_escalate": 0,
        "faithfulness_pass": 0, "faithfulness_total": 0,
        "judge_pass": 0, "judge_total": 0,
    }
    latencies = []

    for i, turn in enumerate(case["turns"]):
        user_text = turn["user"]
        expect = turn.get("expect_decision", "answer")

        start = time.time()
        try:
            result = app.invoke(
                {"customer_query": user_text},
                {"configurable": {"thread_id": tid}},
            )
            error = None
        except Exception as e:  # noqa: BLE001
            result = {}
            error = str(e)
        elapsed = round(time.time() - start, 2)
        latencies.append(elapsed)

        response = str(result.get("response") or "") if not error else ""
        state = result if not error else {}

        # 决策评分
        if error:
            d_pass, d_actual, d_detail = False, "error", f"invoke 异常: {error}"
        else:
            d_pass, d_actual, d_detail = _score_decision(state, expect)
        ct["decision_total"] += 1
        if d_pass:
            ct["decision_pass"] += 1
        else:
            case_ok = False

        # 检索命中判定：从 state.tools_used 推导，无需改 agent。
        # 各业务 agent 检索 miss 时走 _no_answer → tools_used 追加 "{agent}_no_answer_escalation"；
        # 命中并调 LLM 时追加 "{agent}_processing"。两个信号在环外可读。
        tools_used = list(state.get("tools_used") or []) if not error else []
        retrieval_attempted = any(
            "_processing" in t or t.endswith("_no_answer_escalation") for t in tools_used
        )
        retrieval_hit = any("_processing" in t for t in tools_used)
        retrieval_miss = any("_no_answer_escalation" in t for t in tools_used)
        if retrieval_attempted:
            ct["retrieval_attempted"] += 1
            if retrieval_hit:
                ct["retrieval_hit"] += 1

        # 升级决策指标（基于金标期望 vs 实际）：误升级 / 漏升级。
        # 仅对 answer / escalate 两类期望计（refuse 不进入升级指标）。
        if not error and expect in ("answer", "escalate"):
            if expect == "escalate" and d_actual == "escalate":
                ct["correct_escalate"] += 1
            elif expect == "answer" and d_actual == "escalate":
                ct["false_escalate"] += 1
            elif expect == "escalate" and d_actual == "answer":
                ct["miss_escalate"] += 1

        # 答案评分（仅对该轮定义 answer_checks 时）
        answer = {"checked": False, "pass": None, "detail": {}}
        if "answer_checks" in turn and not error:
            checks = dict(turn["answer_checks"])
            # 内部泄漏由下方 no_internal_leak 维度统一检查（正则匹配【XXX's Response】模式）。
            # 不向 must_not_contain 注入 "【"，否则会误拦正文中的【退款政策】等合法用法。
            res = _score_answer_checks(response, checks)
            # A6 引用溯源（本轮 agent 应答必须带引用脚注；升级/拒答轮不适用——
            # tools_used 跨轮累积，升级轮需显式排除 escalated 再判定）
            f_pass, f_detail = _score_faithfulness(
                response, retrieval_hit, bool(state.get("escalated"))
            )
            res["faithfulness"] = f_pass
            res["_faithfulness_detail"] = f_detail
            if retrieval_hit:
                ct["faithfulness_total"] += 1
                if f_pass:
                    ct["faithfulness_pass"] += 1
            # LLM-as-judge 事实一致性（测量增强：抓确定性检查测不到的幻觉/编造）
            j_pass, j_detail = True, "not_applicable"
            if retrieval_hit and not bool(state.get("escalated")):
                _cited = _parse_citations(response)
                _subs = state.get("compound_subs") or []
                if _subs:
                    # A5 复合轮：基座 = 子查询跨域检索（整句 query 在单域检索命中无关条目）
                    _kb_ctx = _compound_kb_context_for_judge(_subs, _cited)
                else:
                    _domain = _AGENT_DISPLAY2DOMAIN.get(state.get("current_agent", ""), "")
                    _kb_ctx = _kb_context_for_judge(_domain, user_text, _cited)
                j_pass, j_detail = _judge_faithfulness(response, _kb_ctx, get_llm())
                if _kb_ctx.strip():
                    ct["judge_total"] += 1
                    if j_pass:
                        ct["judge_pass"] += 1
            res["judge_faithfulness"] = j_pass
            res["_judge_detail"] = j_detail
            a_pass = all(v is True for k, v in res.items() if not k.startswith("_"))
            answer = {"checked": True, "pass": a_pass, "detail": res}
            ct["answer_total"] += 1
            if a_pass:
                ct["answer_pass"] += 1
            else:
                case_ok = False
        elif "answer_checks" in turn and error:
            case_ok = False

        case_turns.append({
            "turn": i + 1,
            "user": user_text,
            "rationale": turn.get("rationale", ""),
            "expect_decision": expect,
            "decision_pass": d_pass,
            "decision_actual": d_actual,
            "decision_detail": d_detail,
            "latency_s": elapsed,
            "query_type": state.get("query_type", ""),
            "confidence": round(float(state.get("query_confidence", 0.0)), 3),
            "current_agent": state.get("current_agent", ""),
            "escalated": bool(state.get("escalated")),
            "retrieval_attempted": retrieval_attempted,
            "retrieval_hit": retrieval_hit,
            "retrieval_miss": retrieval_miss,
            "answer_checked": answer["checked"],
            "answer_pass": answer["pass"],
            "answer_detail": answer["detail"],
            "response_excerpt": response[:200],  # 截断，报告可读
            "response_full": response,  # 完整原文，供审计复现
        })

    report = {
        "id": cid,
        "persona": case["persona"],
        "objective": case["objective"],
        "source": case.get("source", ""),
        "all_pass": case_ok,
        "turns": case_turns,
    }
    return report, ct, latencies


def main():
    # 金标路径可配置（加法，默认不变）：支持留出集/多套金标复用同一评估进程。
    # 留出集验证用法：GOLDEN_PATH=eval/golden_holdout.json BASELINE_FILE=eval/baselines_holdout.json python eval/eval_multi_turn.py
    golden_path = os.environ.get("GOLDEN_PATH") or os.path.join(EVAL_DIR, "golden_multi_turn.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    # 评估正确性：每次运行前清空评估库，保证从零状态开始。
    # 否则 thread_id（eval-{cid}）会被上一次运行的 checkpoint 污染（escalated 残留、persisted_dialogue 混入历史），
    # 导致多轮结果跨运行串扰、不可复现。线上生产库不受影响（进程隔离）。
    if os.path.exists(EVAL_DB):
        os.remove(EVAL_DB)
        print(f"🧹 已清空上次评估库：{EVAL_DB}")

    t0 = time.time()
    reset_token_usage()  # 跑前清零，保证 token/成本可归因到本次运行
    app = make_graph()

    cases_report = []
    totals = {
        "decision_pass": 0, "decision_total": 0,
        "answer_pass": 0, "answer_total": 0,
        "retrieval_attempted": 0, "retrieval_hit": 0,
        "correct_escalate": 0, "false_escalate": 0, "miss_escalate": 0,
        "faithfulness_pass": 0, "faithfulness_total": 0,  # A6 引用溯源
        "judge_pass": 0, "judge_total": 0,                 # LLM-as-judge 事实一致性
    }
    latencies = []

    # 并行执行：case 级并行（每 case 独立 thread_id，多轮上下文在 case 内保持）。
    # 并发安全：token 计数有 _TOKEN_LOCK、API 并发有 _API_SEMAPHORE 门控（MAX_CONCURRENCY=5）、熔断器有锁；
    # reranker 为进程级单例，不随并发增加显存。_EVAL_PARALLEL 默认 4、上限 5，只缩短墙钟不增资源峰值。
    _t_parallel = time.time()
    with ThreadPoolExecutor(max_workers=_EVAL_PARALLEL) as pool:
        futures = {pool.submit(_run_case, app, case): i for i, case in enumerate(golden["cases"])}
        results = [None] * len(futures)  # 按下标回收，保持金标顺序输出
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    print(
        f"⏱ 并行执行 {len(golden['cases'])} cases（并发 {_EVAL_PARALLEL}）"
        f"耗时 {time.time() - _t_parallel:.1f}s"
    )

    for report, ct, lat in results:
        cases_report.append(report)
        for k in totals:  # ct 键集合与 totals 完全一致
            totals[k] += ct[k]
        latencies.extend(lat)

    # -----------------------------------------------------------------------
    # 汇总
    # -----------------------------------------------------------------------
    decision_rate = totals["decision_pass"] / totals["decision_total"] if totals["decision_total"] else 0
    answer_rate = totals["answer_pass"] / totals["answer_total"] if totals["answer_total"] else 0
    all_pass = all(c["all_pass"] for c in cases_report)
    if latencies:
        lat_sorted = sorted(latencies)
        p50 = lat_sorted[len(lat_sorted) // 2]
        p95 = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
    else:
        p50 = p95 = 0

    # 检索命中率：进入检索的轮中命中比例
    retrieval_hit_rate = (
        totals["retrieval_hit"] / totals["retrieval_attempted"]
        if totals["retrieval_attempted"] else 0
    )
    # 升级决策：精确率(该转人工且转对) / 召回率(该转人工都转到)
    tp, fp, fn = totals["correct_escalate"], totals["false_escalate"], totals["miss_escalate"]
    esc_precision = tp / (tp + fp) if (tp + fp) else 0
    esc_recall = tp / (tp + fn) if (tp + fn) else 0

    # 成本基线（方向B：延迟/成本从评估附带沉淀为可对比基线）
    # token 单价做成可配置（元/百万token），qwen-flash 参考价，跨版本主要看 token 数相对变化。
    token_usage = get_token_usage()
    in_price = float(os.environ.get("Eval_TOKEN_IN_PRICE_PER_M", "0.15"))   # 输入 元/百万
    out_price = float(os.environ.get("Eval_TOKEN_OUT_PRICE_PER_M", "1.5"))  # 输出 元/百万
    cost_yuan = round(
        token_usage["prompt_tokens"] / 1_000_000 * in_price
        + token_usage["completion_tokens"] / 1_000_000 * out_price,
        4,
    )

    summary = {
        "name": golden["name"],
        "version": golden["version"],
        "model": os.environ.get("OPENAI_MODEL") or "unknown",  # 实际模型快照，跨版本对比可追溯
        "run_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cases": len(cases_report),
        "turns_total": totals["decision_total"],
        "decision_pass": totals["decision_pass"],
        "decision_total": totals["decision_total"],
        "decision_rate": round(decision_rate, 4),
        "answer_pass": totals["answer_pass"],
        "answer_total": totals["answer_total"],
        "answer_rate": round(answer_rate, 4),
        "retrieval_attempted": totals["retrieval_attempted"],
        "retrieval_hit": totals["retrieval_hit"],
        "retrieval_hit_rate": round(retrieval_hit_rate, 4),
        "faithfulness_pass": totals["faithfulness_pass"],
        "faithfulness_total": totals["faithfulness_total"],
        "faithfulness_rate": round(
            totals["faithfulness_pass"] / totals["faithfulness_total"]
            if totals["faithfulness_total"] else 0, 4
        ),
        "judge_pass": totals["judge_pass"],
        "judge_total": totals["judge_total"],
        "judge_rate": round(
            totals["judge_pass"] / totals["judge_total"]
            if totals["judge_total"] else 0, 4
        ),
        "correct_escalate": totals["correct_escalate"],
        "false_escalate": totals["false_escalate"],
        "miss_escalate": totals["miss_escalate"],
        "escalation_precision": round(esc_precision, 4),
        "escalation_recall": round(esc_recall, 4),
        "all_cases_pass": all_pass,
        "latency_p50_s": round(p50, 2),
        "latency_p95_s": round(p95, 2),
        "token_usage": token_usage,
        "cost_yuan": cost_yuan,
        "cost_pricing": {"in_price_per_m": in_price, "out_price_per_m": out_price, "currency": "CNY"},
        "isolation": {
            "eval_db": EVAL_DB,
            "prod_db": PROD_DB,
            "eval_db_exists": os.path.exists(EVAL_DB),
            "eval_db_size_bytes": os.path.getsize(EVAL_DB) if os.path.exists(EVAL_DB) else 0,
        },
    }

    report = {"summary": summary, "cases": cases_report}
    report_dir = os.path.join(EVAL_DIR, "reports")
    os.makedirs(report_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(report_dir, f"eval_result_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 跨版本基线沉淀（方向B）：本次运行写进 baselines.json，并与上一版对比回归
    prev_baseline = _load_baselines()[-1] if _load_baselines() else None
    _persist_baseline(summary)
    if prev_baseline:
        _report_baseline_diff(prev_baseline, _load_baselines()[-1])
    else:
        print("\n[基线对比] 首次运行，已建立基线（后续版本将自动对比回归）")

    # 失败案例留存（审计文件）：仅收失败轮，含金标期望/理由 + 决策实际 + 完整原文 + 分层 state。
    # 目的：失败可被独立审阅与复现，而非"全绿即自证通过"。
    fail_cases = []
    for c in cases_report:
        for t in c["turns"]:
            failed = (not t["decision_pass"]) or (t["answer_checked"] and not t["answer_pass"])
            if not failed:
                continue
            fail_cases.append({
                "case_id": c["id"],
                "persona": c["persona"],
                "source": c["source"],
                "turn": t["turn"],
                "user": t["user"],
                "rationale": t["rationale"],
                "expect_decision": t["expect_decision"],
                "decision_pass": t["decision_pass"],
                "decision_actual": t["decision_actual"],
                "decision_detail": t["decision_detail"],
                "query_type": t["query_type"],
                "confidence": t["confidence"],
                "current_agent": t["current_agent"],
                "escalated": t["escalated"],
                "answer_checked": t["answer_checked"],
                "answer_pass": t["answer_pass"],
                "answer_detail": t["answer_detail"],
                "response_full": t["response_full"],
            })
    if fail_cases:
        fails_path = os.path.join(report_dir, f"eval_failures_{ts}.json")
        with open(fails_path, "w", encoding="utf-8") as f:
            json.dump({"run_at": summary["run_at"], "fail_count": len(fail_cases), "failures": fail_cases},
                      f, ensure_ascii=False, indent=2)
        print(f"  失败留存：{fails_path}（{len(fail_cases)} 条失败轮，含完整原文供审计）")
    else:
        fails_path = None
        print("  失败留存：本运行无失败轮（全绿）")

    # 控制台摘要
    print("\n" + "=" * 60)
    print(f"多轮金标评估完成：{summary['run_at']}")
    print(f"  决策分(分流正确)：{totals['decision_pass']}/{totals['decision_total']} = {decision_rate:.1%}")
    print(f"  答案分(质量护栏)：{totals['answer_pass']}/{totals['answer_total']} = {answer_rate:.1%}")
    print(f"  检索命中率：{totals['retrieval_hit']}/{totals['retrieval_attempted']} = {retrieval_hit_rate:.1%}")
    print(f"  引用溯源：{totals['faithfulness_pass']}/{totals['faithfulness_total']} = "
          f"{(totals['faithfulness_pass'] / totals['faithfulness_total'] if totals['faithfulness_total'] else 0):.1%}")
    print(f"  judge 事实一致性：{totals['judge_pass']}/{totals['judge_total']} = "
          f"{(totals['judge_pass'] / totals['judge_total'] if totals['judge_total'] else 0):.1%}")
    print(f"  升级决策：精确率={esc_precision:.1%} 召回率={esc_recall:.1%}"
          f"（误升级={totals['false_escalate']} 漏升级={totals['miss_escalate']}）")
    print(f"  全部 case 通过：{'是' if all_pass else '否'}")
    print(f"  延迟：P50={p50:.1f}s  P95={p95:.1f}s")
    print(f"  成本：prompt={token_usage['prompt_tokens']} tok + completion={token_usage['completion_tokens']} tok"
          f" = {token_usage['total_tokens']} tok，约 ¥{cost_yuan}（{token_usage['calls']} 次调用）")
    print(f"  模型：{summary['model']}")
    print(f"  评估库：{EVAL_DB}")
    print(f"  报告已写：{json_path}")
    print("=" * 60)

    # 逐 case 决策 + 答案失败明细（若存在）
    for c in cases_report:
        fails = [t for t in c["turns"] if not t["decision_pass"] or (t["answer_checked"] and not t["answer_pass"])]
        if fails:
            print(f"\n❌ {c['id']} [{c['persona']}]")
            for t in fails:
                print(f"  轮{t['turn']}「{t['user']}」")
                if not t["decision_pass"]:
                    print(f"    ·决策: {t['decision_detail']}")
                if t["answer_checked"] and not t["answer_pass"]:
                    bad = [k for k, v in t["answer_detail"].items() if v is False and not k.startswith('_')]
                    print(f"    ·答案: 未过 {bad}")
                    print(f"    ·回复: {t['response_excerpt']!r}")
    print()


# ---------------------------------------------------------------------------
# 跨版本基线沉淀（方向B：延迟/成本 + 质量指标沉淀为可对比基线，跨版本看回归）
# ---------------------------------------------------------------------------
_BASELINE_FILE = os.environ.get("BASELINE_FILE") or os.path.join(EVAL_DIR, "baselines.json")
# 回归警戒阈值：延迟/成本相对上一版上升超过该比例即标红提醒
_BASELINE_WARN_FACTORS = {
    "latency_p50_s": 1.5,
    "latency_p95_s": 1.5,
    "token_total": 1.3,
}
# 质量指标：下降即提示（不设阈值，任何下降都值得标注）
_BASELINE_QUALITY = [
    "decision_rate", "answer_rate", "retrieval_hit_rate",
    "escalation_precision", "escalation_recall", "faithfulness_rate",
    "judge_rate",
]


def _load_baselines() -> list:
    if not os.path.exists(_BASELINE_FILE):
        return []
    try:
        with open(_BASELINE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _persist_baseline(summary: dict) -> None:
    """把本次运行沉淀为基线记录（追加到 baselines.json，跨版本可对比）。"""
    baselines = _load_baselines()
    baselines.append({
        "version": summary["version"],
        "name": summary["name"],
        "model": summary.get("model", "unknown"),
        "run_at": summary["run_at"],
        "cases": summary["cases"],
        "turns_total": summary["turns_total"],
        "all_cases_pass": summary["all_cases_pass"],
        "decision_rate": summary["decision_rate"],
        "answer_rate": summary["answer_rate"],
        "retrieval_hit_rate": summary["retrieval_hit_rate"],
        "faithfulness_rate": summary.get("faithfulness_rate", 0),
        "judge_rate": summary.get("judge_rate", 0),
        "escalation_precision": summary["escalation_precision"],
        "escalation_recall": summary["escalation_recall"],
        "latency_p50_s": summary["latency_p50_s"],
        "latency_p95_s": summary["latency_p95_s"],
        "token_total": summary["token_usage"]["total_tokens"],
        "cost_yuan": summary["cost_yuan"],
    })
    with open(_BASELINE_FILE, "w", encoding="utf-8") as f:
        json.dump(baselines, f, ensure_ascii=False, indent=2)


def _report_baseline_diff(prev: dict, cur: dict) -> None:
    """与上一版基线对比，输出回归变动（延迟/成本上升、质量下降）。"""
    print("\n[基线对比] 本版 vs 上一版：")
    warns = []
    for key, factor in _BASELINE_WARN_FACTORS.items():
        if prev.get(key) and cur.get(key,) is not None:
            ratio = cur[key] / prev[key]
            flag = " ⚠️ 上升" if ratio >= factor else ""
            print(f"  {key}: {prev[key]} → {cur[key]}（{ratio:.2f}x）{flag}")
            if ratio >= factor:
                warns.append(f"{key} 上升 {ratio:.2f}x")
    for key in _BASELINE_QUALITY:
        if prev.get(key) is not None and cur.get(key) is not None:
            delta = cur[key] - prev[key]
            flag = " ⚠️ 下降" if delta < 0 else ""
            print(f"  {key}: {prev[key]:.3f} → {cur[key]:.3f}（{delta:+.3f}）{flag}")
            if delta < 0:
                warns.append(f"{key} 下降 {delta:.3f}")
    if prev.get("cost_yuan") is not None:
        print(f"  cost_yuan: {prev['cost_yuan']} → {cur['cost_yuan']}（+{cur['cost_yuan'] - prev['cost_yuan']:.4f}）")
    if warns:
        print(f"  → 回归提醒：{'；'.join(warns)}")
    else:
        print("  → 无回归变动（相对上一版基线稳定）")


if __name__ == "__main__":
    main()