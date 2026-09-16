"""测试：结构化分类解析（tools.query_tools.parse_classification）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from tools.query_tools import parse_classification


def test_parse_valid_json():
    r = parse_classification('{"label": "billing", "confidence": 0.92, "complexity": "medium"}')
    assert r.label == "billing"
    assert r.confidence == pytest.approx(0.92)
    assert r.complexity == "medium"


def test_parse_json_with_extra_whitespace():
    r = parse_classification('  {"label":"complaint","confidence":0.8,"complexity":"simple"}  ')
    assert r.label == "complaint"


def test_parse_string_fallback():
    """LLM 输出纯标签字符串 → 降级解析，confidence 保守置 0（触发升级兜底）"""
    r = parse_classification("billing")
    assert r.label == "billing"
    assert r.confidence == 0.0  # fail-safe：不可靠输出置信度归零
    assert r.complexity == "medium"


def test_parse_garbage_input():
    r = parse_classification("帮我写一首诗")
    assert r.label == "general_inquiry"
    assert r.confidence == 0.0


def test_parse_empty_input():
    r = parse_classification("")
    assert r.label == "general_inquiry"
    assert r.confidence == 0.0


def test_parse_invalid_json_with_label_key():
    """JSON 但缺 label → 降级"""
    r = parse_classification('{"confidence": 0.9, "complexity": "simple"}')
    assert r.label == "general_inquiry"
    assert r.confidence == 0.0


def test_parse_json_invalid_confidence():
    """confidence 越界 → Pydantic 拒绝 → 降级路径：label 子串匹配保留，confidence 归零（fail-safe）"""
    r = parse_classification('{"label": "billing", "confidence": 2.5, "complexity": "simple"}')
    # Pydantic 拒绝越界后降级：normalize 子串匹配仍识别出 billing（label 有效）
    assert r.label == "billing"
    # 关键 fail-safe：confidence 归零 → should_escalate(0.0 < 0.6) → 升级兜底（越界高置信不绕过阈值）
    assert r.confidence == 0.0
