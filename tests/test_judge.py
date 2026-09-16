"""测试：LLM-as-judge 事实一致性检查（eval 测量增强）。纯本地，mock LLM，不触发真实 API。

覆盖：judge pass/fail 判定、输出解析失败降级、通道异常降级、开关关闭、KB 上下文提取。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeLLM:
    """最小 mock：invoke 返回 content 或抛异常。"""

    def __init__(self, content=None, exc=None):
        self._content = content
        self._exc = exc
        self.calls = []

    def invoke(self, messages, response_format=None):
        self.calls.append((messages, response_format))
        if self._exc is not None:
            raise self._exc
        return _FakeResp(self._content)


class _FakeResp:
    def __init__(self, content):
        self.content = content


@pytest.fixture(scope="module")
def eval_module():
    """导入 eval 模块（保存/恢复 env 与 cwd，避免污染其他测试）。"""
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


def test_judge_pass(eval_module):
    llm = _FakeLLM(content='{"verdict": "pass", "reason": "回答与引用一致"}')
    ok, detail = eval_module._judge_faithfulness("退货运费商家承担", "【退货运费承担】质量问题退货运费商家承担", llm)
    assert ok is True and "judge=pass" in detail


def test_judge_fail(eval_module):
    llm = _FakeLLM(content='{"verdict": "fail", "reason": "回答承诺 500 元补偿，引用无此数字"}')
    ok, detail = eval_module._judge_faithfulness("可补偿 500 元", "【补偿标准】服务态度类补偿 10-20 元优惠券", llm)
    assert ok is False and "judge=fail" in detail


def test_judge_non_json_fallback(eval_module):
    """裁判输出非 JSON → 解析失败 → 不适用（不误伤答案）。"""
    llm = _FakeLLM(content="抱歉我无法判断")
    ok, detail = eval_module._judge_faithfulness("回答", "【条目】内容", llm)
    assert ok is True and "not_applicable" in detail


def test_judge_channel_exception(eval_module):
    """裁判通道异常 → 不适用（fail-safe）。"""
    llm = _FakeLLM(exc=RuntimeError("通道故障"))
    ok, detail = eval_module._judge_faithfulness("回答", "【条目】内容", llm)
    assert ok is True and "not_applicable" in detail


def test_judge_no_context_not_applicable(eval_module):
    ok, detail = eval_module._judge_faithfulness("回答", "", _FakeLLM(content='{"verdict": "fail"}'))
    assert ok is True and detail == "not_applicable"


def test_judge_disabled(eval_module):
    """开关关闭 → 不适用（模拟 JUDGE_ENABLED=0）。"""
    saved = eval_module._JUDGE_ENABLED
    eval_module._JUDGE_ENABLED = False
    try:
        ok, detail = eval_module._judge_faithfulness("回答", "【条目】内容", _FakeLLM(content='{"verdict": "fail"}'))
        assert ok is True and detail == "not_applicable"
    finally:
        eval_module._JUDGE_ENABLED = saved


def test_kb_context_from_titles(eval_module):
    """按引用标题取 KB 条目内容（真实 KB，title 唯一跨域搜）。"""
    ctx = eval_module._kb_context_from_titles(["退货运费承担"])
    assert "退货运费" in ctx or "运费" in ctx
    assert eval_module._kb_context_from_titles([]) == ""
    assert eval_module._kb_context_from_titles(["不存在的标题xyz"]) == ""


def test_kb_context_for_judge_uses_domain_retrieval(eval_module):
    """核对基座优先用域检索全文（agent 实际注入的上下文），空则回落引用标题。"""
    ctx = eval_module._kb_context_for_judge("billing", "退货运费谁出", ["退货运费承担"])
    assert "运费" in ctx or "退货" in ctx
    # 域检索为空（A4 改写重检路径）→ 回落引用 titles
    ctx2 = eval_module._kb_context_for_judge("", "", ["退货运费承担"])
    assert "运费" in ctx2
    assert eval_module._kb_context_for_judge("", "", []) == ""
