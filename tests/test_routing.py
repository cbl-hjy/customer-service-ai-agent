"""测试：升级决策硬边界（routing.should_escalate）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from routing import LABEL_DEFAULT_COMPLEXITY, should_escalate, wants_human


@pytest.mark.parametrize(
    "label, confidence, complexity, expect_escalate",
    [
        # 正常高置信 → 不升级（一线处理）
        ("product_info", 0.98, "simple", False),
        ("billing", 0.95, "simple", False),
        ("general_inquiry", 0.90, "simple", False),
        # 低置信 → 升级（不确定不硬答）
        ("billing", 0.30, "simple", True),
        ("product_info", 0.59, "medium", True),
        # 复杂 → 升级
        ("technical_support", 0.80, "complex", True),
        # 投诉 → 升级（人工跟进）
        ("complaint", 0.90, "simple", True),
        # 护栏优先：out_of_scope 不升级（直接拒绝）
        ("out_of_scope", 1.0, "simple", False),
        ("out_of_scope", 0.10, "simple", False),
        # 降级兜底：confidence=0 → 升级（fail-safe）
        ("general_inquiry", 0.0, "medium", True),
    ],
)
def test_should_escalate(label, confidence, complexity, expect_escalate):
    escalated, reason = should_escalate(label, confidence, complexity)
    assert escalated == expect_escalate
    # 升级必有原因（fail-safe 可追溯）
    if expect_escalate:
        assert reason, "升级必须带原因"


def test_out_of_scope_never_escalates():
    """负向边界：out_of_scope 走护栏拒绝，绝不进入升级流程（护栏不被削弱）"""
    for conf in (0.0, 0.5, 1.0):
        escalated, reason = should_escalate("out_of_scope", conf, "complex")
        assert escalated is False
        assert "护栏" in reason


def test_label_default_complexity_cover_all_labels():
    """兜底表必须覆盖全部 6 个标签（harness 兜底完整性）"""
    from tools.query_tools import _CLASS_LABELS

    for label in _CLASS_LABELS:
        assert label in LABEL_DEFAULT_COMPLEXITY, f"缺少兜底复杂度: {label}"
        assert LABEL_DEFAULT_COMPLEXITY[label] in ("simple", "medium", "complex")


@pytest.mark.parametrize(
    "query, expect",
    [
        # v15 真问题：用户主动求人工 → 必须识别
        ("怎么联系人工客服？有严重问题当面说", True),
        ("我要找经理投诉", True),
        ("帮我转人工处理", True),
        ("请转接人工坐席", True),
        ("有没有真人客服", True),
        # 正常查询 → 不误伤
        ("你们手机多少钱", False),
        ("人工智能怎么用", False),  # 裸"人工"子串不误伤
        ("退款什么时候到账", False),
        ("", False),
    ],
)
def test_wants_human(query, expect):
    assert wants_human(query) is expect
