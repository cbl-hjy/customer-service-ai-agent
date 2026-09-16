"""测试：A5 查询分解（复合检测 + 子查询拆解解析）。纯本地，不触发 LLM。

覆盖：
  1. detect_compound：多问句/并列连词+多域共现命中；单诉求/闲聊不误判
  2. parse_decomposed：JSON 数组/对象包装/非法输入/域规范化 fail-safe
  3. 拆解器注入/清除配对（与 A4 改写器同纪律）
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tools.query_decompose as _qd
from tools.query_decompose import clear_query_decomposer, detect_compound, parse_decomposed, set_query_decomposer


@pytest.fixture(autouse=True)
def _clear_decomposer():
    """隔离拆解器注入（与 test_kb_retriever 同纪律）。"""
    clear_query_decomposer()
    yield
    clear_query_decomposer()


# ---------------------------------------------------------------------------
# 1) detect_compound：复合检测
# ---------------------------------------------------------------------------
def test_compound_two_questions():
    """多问句 = 复合强信号（mt-402 场景）。"""
    assert detect_compound("收到的手机屏幕碎了，能换货吗？能给点补偿吗？")


def test_compound_conjunction_multi_domain():
    """单句内并列连词 + 多域共现 = 复合（mt-401 场景：退货 billing + 发票 billing）。"""
    # 退货(billing) + 发票(billing)：同域双诉求，靠多问句信号（"运费谁出？...还能补开吗？"）
    assert detect_compound("耳机有质量问题想退货，运费谁出？退货后发票还能补开吗？")


def test_compound_three_part():
    """三问句复合（mt-401 精简）。"""
    assert detect_compound("能退货吗？运费谁出？发票能补开吗？")


def test_not_compound_single_question():
    """单诉求不判复合。"""
    assert not detect_compound("能换货吗")
    assert not detect_compound("这个耳机多少钱")


def test_not_compound_statement():
    """陈述句（投诉/报修）不判复合——复合检测只认多诉求，不抢分类器活。"""
    assert not detect_compound("收到的耳机有瑕疵，要求补偿")
    assert not detect_compound("电脑开不了机怎么回事")
    assert not detect_compound("帮我写一首关于夏天的诗")


def test_not_compound_conjunction_single_domain():
    """单域并列连词不判复合（"优惠券和赠品"同域）。"""
    assert not detect_compound("有优惠券吗，赠品有什么")


def test_compound_empty_query():
    assert not detect_compound("")
    assert not detect_compound(None)


def test_compound_mt404():
    """mt-404：预售发货+缺货退款（多问句）。"""
    assert detect_compound("预售的耳机多久发货？要是等太久能退款吗？")


def test_compound_mt403():
    """mt-403：分期+发票（多问句）。"""
    assert detect_compound("这台电脑能分期付款吗？分期的话发票怎么开？")


# ---------------------------------------------------------------------------
# 2) parse_decomposed：拆解输出解析
# ---------------------------------------------------------------------------
def test_parse_array():
    raw = '[{"sub_query": "能换货吗", "domain": "general"}, {"sub_query": "能给点补偿吗", "domain": "complaint"}]'
    subs = parse_decomposed(raw)
    assert len(subs) == 2
    assert subs[0] == {"sub_query": "能换货吗", "domain": "general"}
    assert subs[1] == {"sub_query": "能给点补偿吗", "domain": "complaint"}


def test_parse_wrapped_object():
    raw = '{"sub_queries": [{"sub_query": "能退货吗", "domain": "billing"}, {"sub_query": "运费谁出", "domain": "billing"}]}'
    subs = parse_decomposed(raw)
    assert len(subs) == 2
    assert subs[0]["domain"] == "billing"


def test_parse_domain_normalize():
    """域别名/大小写/非法域 → 规范化（非法域回落 general）。"""
    raw = '[{"sub_query": "a", "domain": "General_Inquiry"}, {"sub_query": "b", "domain": "product_info"}, {"sub_query": "c", "domain": "火星"}]'
    subs = parse_decomposed(raw)
    assert subs[0]["domain"] == "general"
    assert subs[1]["domain"] == "product"
    assert subs[2]["domain"] == "general"  # 非法域兜底


def test_parse_invalid():
    assert parse_decomposed("") == []
    assert parse_decomposed("not json") == []
    assert parse_decomposed('{"foo": 1}') == []  # 无 sub_queries 键
    assert parse_decomposed('[{"sub_query": "   "}]') == []  # 空子查询丢弃
    assert parse_decomposed(None) == []


def test_parse_single():
    """单子查询（未复合但被误调拆解时兜底）。"""
    subs = parse_decomposed('[{"sub_query": "能换货吗", "domain": "general"}]')
    assert len(subs) == 1


# ---------------------------------------------------------------------------
# 3) 拆解器注入/清除配对
# ---------------------------------------------------------------------------
def test_decomposer_inject_clear():
    assert _qd.get_query_decomposer() is None
    set_query_decomposer(lambda q: "[]")
    assert _qd.get_query_decomposer() is not None
    clear_query_decomposer()
    assert _qd.get_query_decomposer() is None
