"""测试：A6 引用溯源（harness 附加引用 + eval 引用真实性检查）。纯本地，不触发 LLM。

覆盖三层：
  1. kb_retriever.retrieve_titles：环外取命中标题（与 retrieve 同判定路径）
  2. multi_agent_customer_service._attach_citations：harness 附加"参考来源"脚注
  3. eval_multi_turn._parse_citations / _score_faithfulness：引用解析 + 真实性检查
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kb_retriever as _kbr
from kb_retriever import retrieve_titles

# 引入 harness 模块（不触发图构建/LLM，仅取纯函数与常量）
import multi_agent_customer_service as _mcs


@pytest.fixture(autouse=True)
def _pure_retrieval():
    """隔离 A4 改写器（与 test_kb_retriever 同纪律）：本文件断言纯检索行为。"""
    _kbr.clear_query_rewriter()
    yield
    _kbr.clear_query_rewriter()


# ---------------------------------------------------------------------------
# 1) retrieve_titles：环外取命中标题
# ---------------------------------------------------------------------------
def test_retrieve_titles_hit():
    titles = retrieve_titles("product", "你们有游戏本卖吗", top_k=3)
    assert isinstance(titles, list) and titles
    # 与 retrieve() 命中同一域：标题应属于游戏本/电脑域条目
    joined = " ".join(titles)
    assert "游戏本" in joined or "游戏" in joined or "电脑" in joined


def test_retrieve_titles_top1_matches_top_title():
    t1 = _kbr.get_retriever().retrieve_top_title("product", "你们有游戏本卖吗")
    titles = retrieve_titles("product", "你们有游戏本卖吗", top_k=1)
    assert titles and titles[0] == t1


def test_retrieve_titles_ood_empty():
    assert retrieve_titles("product", "你们有火星车卖吗") == []
    assert retrieve_titles("complaint", "太空飞船怎么修") == []


# ---------------------------------------------------------------------------
# 2) _attach_citations：harness 统一附加引用
# ---------------------------------------------------------------------------
def test_attach_citations_appends_footer(monkeypatch):
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["手机", "平板"])
    state = {
        "customer_query": "X1 Pro 怎么样",
        "response": "这是很好的手机。",
        "tools_used": [],
    }
    _mcs._attach_citations("product_agent", state)
    assert state["response"].endswith("\n\n参考来源：手机、平板")
    assert "citation" in state["tools_used"]


def test_attach_citations_no_hit_no_footer(monkeypatch):
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: [])
    state = {"customer_query": "火星车", "response": "回答", "tools_used": []}
    _mcs._attach_citations("product_agent", state)
    assert state["response"] == "回答"
    assert "citation" not in state["tools_used"]


def test_attach_citations_unknown_agent_no_op(monkeypatch):
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["手机"])
    state = {"customer_query": "q", "response": "r", "tools_used": []}
    _mcs._attach_citations("外星专家", state)
    assert state["response"] == "r"


def test_attach_citations_agent_key_mapping():
    """create_agent_node 传入的是 agent key（product_agent 等），必须映射到正确 domain。"""
    assert _mcs._AGENT_DOMAIN_MAP["product_agent"] == "product"
    assert _mcs._AGENT_DOMAIN_MAP["tech_agent"] == "tech"
    assert _mcs._AGENT_DOMAIN_MAP["billing_agent"] == "billing"
    assert _mcs._AGENT_DOMAIN_MAP["complaint_agent"] == "complaint"
    assert _mcs._AGENT_DOMAIN_MAP["general_agent"] == "general"


def test_attach_citations_disabled_by_env(monkeypatch):
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["手机"])
    # 通过重置模块常量模拟 ENABLE_CITATIONS=False（环境变量在 import 时已求值，
    # 直接改模块属性是等效的最小改动；产品行为开关逻辑见 _attach_citations 首行）
    # 2026-08-18 拆包：开关由 citations 模块持有，patch 其定义处（壳 re-export 是值拷贝）
    import citations as _cit
    saved = _cit.ENABLE_CITATIONS
    _cit.ENABLE_CITATIONS = False
    try:
        state = {"customer_query": "q", "response": "r", "tools_used": []}
        _mcs._attach_citations("product_agent", state)
        assert state["response"] == "r"
    finally:
        _cit.ENABLE_CITATIONS = saved


# ---------------------------------------------------------------------------
# 2b) _attach_citations 多轮适配（2026-09-16，persona-correct 脚注失真教训）：
#     指代/省略追问轮 → 前序轮引用条目并入脚注（本轮在前、prior 追加、去重保序）
# ---------------------------------------------------------------------------
def _mk_dialogue(*turns):
    """构造 persisted_dialogue：(is_user, content) 交替轮次。"""
    return [{"content": c, "is_user": u, "timestamp": "t"} for u, c in turns]


def test_followup_query_merges_prior_citations(monkeypatch):
    """C2 追问轮（含指代词）：前序轮脚注条目并入，本轮在前 prior 追加。"""
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["错发处理"])
    state = {
        "customer_query": "不是，我根本还没买，就是先问问",
        "response": "价保需签收后 7 天内提出。",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "X2 Max 有价保吗"),
            (False, "价保规则说明。\n\n参考来源：价保申请流程、退款政策"),
        ),
    }
    _mcs._attach_citations("general_agent", state)
    assert state["response"].endswith(
        "\n\n参考来源：错发处理、价保申请流程、退款政策"
    )


def test_followup_short_query_merges_prior(monkeypatch):
    """C2b 短 query 追问（≤12 字无指代词同样触发）。"""
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: [])
    state = {
        "customer_query": "耳机呢",
        "response": "耳机选购说明。",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "推荐个手机"),
            (False, "手机推荐。\n\n参考来源：3000元内手机推荐"),
        ),
    }
    # 追问轮检索 miss 但 prior 存在 → 脚注仍附加（真实来源不缺失）
    _mcs._attach_citations("general_agent", state)
    assert state["response"].endswith("\n\n参考来源：3000元内手机推荐")


def test_new_topic_long_query_no_merge(monkeypatch):
    """C3 非追问轮（长 query 无指代词）：只标本轮检索，前序脚注不并入。"""
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["退货政策"])
    state = {
        "customer_query": "顺便问一下你们的退货政策是怎么规定的",
        "response": "退货政策说明。",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "X1 Pro 怎么样"),
            (False, "手机说明。\n\n参考来源：X1 Pro 规格"),
        ),
    }
    _mcs._attach_citations("general_agent", state)
    assert state["response"].endswith("\n\n参考来源：退货政策")


def test_first_turn_no_merge(monkeypatch):
    """C1 单轮（无前序 assistant 轮）：行为与原版一致。"""
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["手机"])
    state = {
        "customer_query": "X1 Pro 怎么样",
        "response": "很好的手机。",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue((True, "X1 Pro 怎么样")),
    }
    _mcs._attach_citations("product_agent", state)
    assert state["response"].endswith("\n\n参考来源：手机")


def test_prior_overlap_dedup(monkeypatch):
    """C5 prior 与本轮检索重叠：去重不重复标注。"""
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["价保申请流程", "新条目"])
    state = {
        "customer_query": "那具体的申请流程呢",
        "response": "流程说明。",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "价保政策"),
            (False, "价保。\n\n参考来源：价保申请流程"),
        ),
    }
    _mcs._attach_citations("general_agent", state)
    assert state["response"].endswith("\n\n参考来源：价保申请流程、新条目")


def test_prior_no_footer_history_safe(monkeypatch):
    """C4 前序 assistant 消息无脚注：不崩，退化为本轮检索。"""
    monkeypatch.setattr(_kbr, "retrieve_titles", lambda domain, query, top_k=3: ["手机"])
    state = {
        "customer_query": "那电池呢",
        "response": "电池说明。",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "手机怎么样"),
            (False, "很好用。"),  # 无脚注（如升级轮/工具轮）
        ),
    }
    _mcs._attach_citations("product_agent", state)
    assert state["response"].endswith("\n\n参考来源：手机")


# ---------------------------------------------------------------------------
# 2c) _prior_kb_fallback：追问轮检索 miss 的前序轮知识回退
#     （检索注入多轮化方案 E，2026-09-16）——注入侧与脚注侧（2b）同源闭环
# ---------------------------------------------------------------------------
class _FakeRetriever:
    def __init__(self, by_title):
        self._data = {"product": [{"title": t, "content": c} for t, c in by_title.items()]}


def test_f1_followup_miss_fallback_returns_context(monkeypatch):
    """F1 追问轮 + prior 脚注 → 返回条目全文并打 prior_kb_fallback 标记。"""
    monkeypatch.setattr(_kbr, "get_retriever",
                        lambda: _FakeRetriever({"价保申请流程": "签收后 7 天内可申请价保。"}))
    from multi_agents.product_agent import ProductAgent
    agent = ProductAgent()
    state = {
        "customer_query": "不是，我根本还没买",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "X2 Max 有价保吗"),
            (False, "价保规则。\n\n参考来源：价保申请流程"),
        ),
    }
    ctx = agent._prior_kb_fallback(state)
    assert "价保申请流程" in ctx and "签收后 7 天内" in ctx
    assert "prior_kb_fallback" in state["tools_used"]


def test_f2_new_topic_no_fallback(monkeypatch):
    """F2 非追问轮（长 query 无指代词）：回退不触发（退回升级路径）。"""
    monkeypatch.setattr(_kbr, "get_retriever", lambda: _FakeRetriever({"A": "a"}))
    from multi_agents.product_agent import ProductAgent
    agent = ProductAgent()
    state = {
        "customer_query": "顺便问一下你们的退货政策是怎么规定的",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "手机怎么样"),
            (False, "手机。\n\n参考来源：A"),
        ),
    }
    assert agent._prior_kb_fallback(state) == ""
    assert "prior_kb_fallback" not in state["tools_used"]


def test_f3_first_turn_no_fallback():
    """F3 单轮（无前序 assistant 轮）：回退不触发——单轮行为零变化。"""
    from multi_agents.product_agent import ProductAgent
    agent = ProductAgent()
    state = {
        "customer_query": "耳机呢",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue((True, "耳机呢")),
    }
    assert agent._prior_kb_fallback(state) == ""


def test_f6_agent_process_uses_fallback_not_escalate(monkeypatch):
    """F6 集成：ProductAgent 追问轮 miss → prior 注入走正常应答（不升级）。"""
    monkeypatch.setattr(_kbr, "get_retriever",
                        lambda: _FakeRetriever({"价保申请流程": "签收后 7 天内可申请价保。"}))
    from multi_agents import product_agent as _pa

    captured = {}

    class _FakeResp:
        content = "还没买的话，下单后签收 7 天内可以申请价保退差价。"

    def _fake_match(query):
        return ""  # 模拟本轮检索 miss

    def _fake_call_llm(llm, messages, **kw):
        captured["prompt"] = "\n".join(str(getattr(m, "content", m)) for m in messages)
        return _FakeResp()

    monkeypatch.setattr(_pa.ProductAgent, "_match_products", lambda self, q: _fake_match(q))
    monkeypatch.setattr(_pa, "call_llm", _fake_call_llm)

    agent = _pa.ProductAgent()
    state = {
        "customer_query": "不是，我根本还没买",
        "session_id": "t-f6",
        "tools_used": [],
        "persisted_dialogue": _mk_dialogue(
            (True, "X2 Max 有价保吗"),
            (False, "价保规则。\n\n参考来源：价保申请流程"),
        ),
    }
    result = agent.process(state)
    assert not result.get("escalated"), "追问轮 miss 不应升级"
    assert "签收 7 天内" in result["response"]
    # 注入上下文含前序轮条目全文（检索注入多轮化）
    assert "价保申请流程" in captured["prompt"] and "签收后 7 天内" in captured["prompt"]


# ---------------------------------------------------------------------------
# 3) eval 引用解析 + faithfulness（导入 eval 模块需保存/恢复 env 与 cwd）
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eval_module():
    import importlib
    old_db = os.environ.get("CHECKPOINT_DB_PATH")
    old_cwd = os.getcwd()
    try:
        mod = importlib.import_module("eval.eval_multi_turn")
        yield mod
    finally:
        os.environ.pop("CHECKPOINT_DB_PATH", None)
        if old_db:
            os.environ["CHECKPOINT_DB_PATH"] = old_db
        os.chdir(old_cwd)


def test_parse_citations(eval_module):
    resp = "本机支持快充。\n\n参考来源：手机快充、手机电池"
    assert eval_module._parse_citations(resp) == ["手机快充", "手机电池"]
    assert eval_module._parse_citations("没有引用") == []


def test_score_faithfulness(eval_module):
    # 检索命中 + 带引用 → 通过
    ok, _ = eval_module._score_faithfulness("好手机。\n\n参考来源：手机", True, False)
    assert ok is True
    # 检索命中 + 无引用 → 失败
    ok, detail = eval_module._score_faithfulness("好手机。", True, False)
    assert ok is False and "未标注引用" in detail
    # 检索未命中（升级/拒答）→ 不适用
    ok, detail = eval_module._score_faithfulness("抱歉已转人工", False, False)
    assert ok is True and detail == "not_applicable"
    # 升级轮（即使 tools_used 残留上轮检索信号）→ 不适用
    ok, detail = eval_module._score_faithfulness("您的工单已升级至人工客服处理", True, True)
    assert ok is True and detail == "not_applicable"
