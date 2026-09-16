"""测试：产品库型号匹配（v02 修复——消除比价/选型过度升级）"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from multi_agents.product_agent import ProductAgent


@pytest.fixture
def agent():
    return ProductAgent()


def test_match_by_product_name(agent):
    """按产品名精确匹配（原有行为）"""
    result = agent._match_products("我想买手机")
    assert "手机" in result


def test_match_by_model_name(agent):
    """kb-v2 型号直接命中（v1 的 X3 Lite/X2 Max 在 v2 已不存在，改用 v2 型号比价）"""
    result = agent._match_products("星环 X1 Pro 和 Book Air 13 比哪个性价比高")
    assert result, "型号应命中产品库，不应无匹配"
    assert "手机" in result


def test_match_other_models(agent):
    """其他型号同样命中（kb-v2 用星环 Book 系列替代 v1 的 ThinkPad）"""
    result = agent._match_products("星环 Book Pro 14 多少钱")
    assert result
    assert "Book Pro" in result


def test_no_match_returns_empty(agent):
    """库外产品 → 空串（触发 _no_answer 升级，诚实不编造）"""
    result = agent._match_products("你们有火星车卖吗")
    assert result == ""
