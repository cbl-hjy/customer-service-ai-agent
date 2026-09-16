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
