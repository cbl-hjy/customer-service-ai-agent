"""测试：A5 复合查询处理节点（compound_node）与触发策略。纯本地，mock LLM 不耗 API。

覆盖：
  1. _try_decompose：注入拆解器命中 / 未注入回退 / 拆解失败回退
  2. compound_node：全子查询命中 → 综合回答 + 引用溯源；任一 miss → 升级兜底
  3. classify 触发策略：仅"原路径会升级"才尝试复合（mt-402 模式）
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kb_retriever as _kbr
import multi_agent_customer_service as _mcs
import nodes.classify as _nc
import nodes.compound as _ncomp
import tools.query_decompose as _qd

from tools.query_decompose import clear_query_decomposer, set_query_decomposer


def _patch_llm(monkeypatch, llm):
    """2026-08-18 拆包：patch 使用处（classify/compound 各持有 get_llm 引用，
    壳 re-export 是绑定拷贝，patch 壳无效——mock 必须打在定义模块上）。"""
    monkeypatch.setattr(_nc, "get_llm", lambda: llm)
    monkeypatch.setattr(_ncomp, "get_llm", lambda: llm)


@pytest.fixture(autouse=True)
def _pure_retrieval():
    """隔离 A4 改写器 + A5 拆解器（与既有测试同纪律）：纯检索口径。"""
    _kbr.clear_query_rewriter()
    clear_query_decomposer()
    yield
    _kbr.clear_query_rewriter()
    clear_query_decomposer()


class _FakeResp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, resp: str):
        self._resp = resp
        self.calls = []

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _FakeResp(self._resp)


# ---------------------------------------------------------------------------
# 1) _try_decompose：触发与回退
# ---------------------------------------------------------------------------
def test_try_decompose_hit(monkeypatch):
    """注入拆解器 + 复合查询 → 返回子查询列表。"""
    set_query_decomposer(lambda q: '{"sub_queries": [{"sub_query": "能换货吗", "domain": "general"}, {"sub_query": "能给点补偿吗", "domain": "complaint"}]}')
    subs = _mcs._try_decompose("能换货吗？能给点补偿吗？")
    assert len(subs) == 2
    assert subs[0]["domain"] == "general"
    assert subs[1]["domain"] == "complaint"


def test_try_decompose_no_inject():
    """拆解器未注入（评估纯检索口径）→ 返回 []（行为零变化）。"""
    assert _mcs._try_decompose("能换货吗？能给点补偿吗？") == []


def test_try_decompose_single_question():
    """非复合查询（单问句）不触发拆解（即使注入拆解器）。"""
    set_query_decomposer(lambda q: '{"sub_queries": [{"sub_query": "能换货吗", "domain": "general"}]}')
    # detect_compound 未命中 → 不调拆解器
    assert _mcs._try_decompose("能换货吗") == []


def test_try_decompose_bad_output():
    """拆解器返回非法 JSON → 回退 []。"""
    set_query_decomposer(lambda q: "not json")
    assert _mcs._try_decompose("能换货吗？能给点补偿吗？") == []


def test_try_decompose_single_subquery():
    """拆解结果只有 1 个子查询（模型认为非复合）→ 回退 []。"""
    set_query_decomposer(lambda q: '{"sub_queries": [{"sub_query": "能换货吗", "domain": "general"}]}')
    assert _mcs._try_decompose("能换货吗？能给点补偿吗？") == []


# ---------------------------------------------------------------------------
# 2) compound_node：全命中综合回答 / miss 升级兜底
# ---------------------------------------------------------------------------
def test_compound_node_all_hit(monkeypatch):
    """全部子查询检索命中 → 综合回答 + 引用脚注 + compound_processing 信号。"""
    llm = _FakeLLM("可以换货，质量问题可补偿。")
    _patch_llm(monkeypatch, llm)
    state = {
        "customer_query": "能换货吗？能给点补偿吗？",
        "compound_subs": [
            {"sub_query": "能换货吗", "domain": "general"},
            {"sub_query": "能给点补偿吗", "domain": "complaint"},
        ],
        "tools_used": ["query_classification"],
        "persisted_dialogue": [],
        "session_id": "t-compound",
    }
    out = _mcs.compound_node(state)
    assert "换货" in out["response"]
    assert "参考来源：" in out["response"]
    assert any("_processing" in t for t in out["tools_used"])
    assert "citation" in out["tools_used"]
    assert not out.get("escalated")
    # 确认 LLM 收到了两个子诉求的 KB 上下文
    joined = "".join(str(m.content) for m in llm.calls[0][0])
    assert "子诉求" in joined


def test_compound_node_miss_escalate(monkeypatch):
    """任一子查询检索 miss → 升级兜底（不编造），走 escalate 路径。"""
    llm = _FakeLLM("不应被调用")
    _patch_llm(monkeypatch, llm)
    state = {
        "customer_query": "能换货吗？能给点火星车吗？",
        "compound_subs": [
            {"sub_query": "能换货吗", "domain": "general"},
            {"sub_query": "能发火星车吗", "domain": "product"},
        ],
        "tools_used": ["query_classification"],
        "persisted_dialogue": [],
        "session_id": "t-compound-miss",
    }
    out = _mcs.compound_node(state)
    assert out.get("escalated") is True
    assert "人工" in out["escalation_reason"]
    assert llm.calls == []  # miss → 不调 LLM 生成（不编造）


def test_compound_node_llm_fail(monkeypatch):
    """LLM 生成异常 → 升级兜底。"""
    def boom(*a, **k):
        raise RuntimeError("LLM down")
    _patch_llm(monkeypatch, boom)
    state = {
        "customer_query": "能换货吗？能给点补偿吗？",
        "compound_subs": [
            {"sub_query": "能换货吗", "domain": "general"},
            {"sub_query": "能给点补偿吗", "domain": "complaint"},
        ],
        "tools_used": ["query_classification"],
        "persisted_dialogue": [],
        "session_id": "t-compound-llm",
    }
    out = _mcs.compound_node(state)
    assert out.get("escalated") is True


# ---------------------------------------------------------------------------
# 3) classify 触发策略：仅原路径升级时尝试复合
# ---------------------------------------------------------------------------
def _run_classify(query, monkeypatch, decomposer_json=None, label="complaint", conf=0.95, complexity="complex"):
    """跑 classify_query_node（绕过 LLM 分类：直接 mock）。"""
    from langchain_core.messages import SystemMessage
    if decomposer_json:
        set_query_decomposer(lambda q: decomposer_json)
    # mock 分类器输出
    _cls = _FakeLLM(f'{{"label": "{label}", "confidence": {conf}, "complexity": "{complexity}"}}')
    _patch_llm(monkeypatch, _cls)
    state = {
        "customer_query": query,
        "session_id": "t-cls",
        "tools_used": [],
        "conversation_history": [],
        "persisted_dialogue": [],
        "dialogue_summary": "",
        "next_agent": "",
        "messages": [],
        "query_confidence": 0.0,
        "query_complexity": "medium",
        "escalated": False,
        "escalation_reason": "",
        "escalation_summary": "",
    }
    return _mcs.classify_query_node(state)


def test_classify_compound_only_when_escalate(monkeypatch):
    """原路径升级（complaint）→ 尝试复合 → 拆解成功则取消升级。"""
    out = _run_classify(
        "收到的手机屏幕碎了，能换货吗？能给点补偿吗？",
        monkeypatch,
        decomposer_json='{"sub_queries": [{"sub_query": "能换货吗", "domain": "general"}, {"sub_query": "能给点补偿吗", "domain": "complaint"}]}',
    )
    assert out["compound_subs"]  # 复合拆解命中
    assert out["escalated"] is False  # 不再误升级


def test_classify_no_escalate_keeps_path(monkeypatch):
    """原路径不升级（billing）→ 不尝试复合，行为不变。"""
    out = _run_classify(
        "耳机有质量问题想退货，运费谁出？退货后发票还能补开吗？",
        monkeypatch,
        label="billing", complexity="medium",
    )
    assert out["compound_subs"] == []
    assert out["escalated"] is False


def test_classify_escalate_no_decomposer(monkeypatch):
    """拆解器未注入（纯检索口径）→ 复合查询仍升级（fail-safe，不改变原行为）。"""
    out = _run_classify(
        "收到的手机屏幕碎了，能换货吗？能给点补偿吗？",
        monkeypatch,
    )
    assert out["compound_subs"] == []
    assert out["escalated"] is True


def test_classify_escalate_single_complaint(monkeypatch):
    """单诉求投诉（mt-308 模式）→ 无复合信号 → 正常升级。"""
    out = _run_classify(
        "收到的耳机有瑕疵，要求补偿",
        monkeypatch,
        decomposer_json='{"sub_queries": [{"sub_query": "要求补偿", "domain": "complaint"}]}',
    )
    assert out["compound_subs"] == []  # 单诉求不触发拆解
    assert out["escalated"] is True
