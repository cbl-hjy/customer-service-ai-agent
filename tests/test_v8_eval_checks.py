"""V8 测试：eval 检查器 no_system_error 规则细化（明确故障词 vs 裸"系统"子串）。

背景（审查清单 V8）：mt-201 实证——正常回复含"系统"二字曾被误判失败（回归信号污染）。
修复：匹配"系统错误/技术问题/智能客服服务暂时不可用"等明确故障词，而非裸"系统"子串；
补齐 product_agent 兜底文案"技术问题"（其余 4 agent 用"系统错误"，此前漏网）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_multi_turn import (
    _no_system_error,
    _SYSTEM_ERROR_TOKENS,
    _score_answer_checks,
)


# ---------------------------------------------------------------------------
# 明确故障词命中（应判失败 = no_system_error False）
# ---------------------------------------------------------------------------

def test_system_error_token_caught():
    """业务 agent 兜底文案（4 个 agent 用"系统错误"）必须被抓住。"""
    replies = [
        "抱歉，处理您的账单问题时遇到系统错误，请稍后重试。",
        "抱歉，处理您的投诉时遇到系统错误，请稍后重试。",
        "抱歉，处理您的咨询时遇到系统错误，请稍后重试。",
        "抱歉，处理您的技术问题时遇到系统错误，请稍后重试。",
    ]
    for r in replies:
        assert _no_system_error(r) is False, f"系统错误兜底未抓住: {r}"


def test_tech_problem_token_caught():
    """V8 新增：product_agent 兜底文案"技术问题"必须被抓住（此前漏网）。"""
    r = "抱歉，处理您的产品查询时遇到技术问题，请稍后重试。"
    assert _no_system_error(r) is False, "技术问题兜底未抓住（V8 修复失效）"


def test_service_unavailable_caught():
    """熔断降级提示（403/额度耗尽）必须被抓住。"""
    r = "抱歉，智能客服服务暂时不可用（可能是服务限流或额度已用尽），请稍候片刻再试。"
    assert _no_system_error(r) is False


def test_error_prefix_caught():
    """错误前缀类兜底必须被抓住。"""
    assert _no_system_error("Error: Agent product_agent not found") is False
    assert _no_system_error("Error: No response from agent") is False
    assert _no_system_error("未获取到回复") is False


# ---------------------------------------------------------------------------
# 正常回复不误报（V8 核心：裸"系统"二字不再误判）
# ---------------------------------------------------------------------------

def test_bare_system_word_not_false_positive():
    """正常回复含"系统"二字（降噪系统/操作系统/系统提示）不误判为故障。"""
    replies = [
        "这款耳机搭载先进的主动降噪系统，能有效过滤环境噪音。",
        "手机操作系统为 Android 15，支持最新应用。",
        "系统提示：您的订单将在 3 个工作日内送达。",
    ]
    for r in replies:
        assert _no_system_error(r) is True, f"裸系统二字误报: {r}"


def test_normal_answers_pass_all_checks():
    """正常回复通过 _score_answer_checks 全部质量检查（含 no_system_error）。"""
    r = "质量问题退货的运费由商家承担，请放心。退款一般在 3-5 个工作日到账。"
    results = _score_answer_checks(r, {"must_contain": ["运费", "承担"]})
    assert results["no_system_error"] is True
    assert results["must_contain"] is True
    assert results["non_empty"] is True


# ---------------------------------------------------------------------------
# token 列表完整性（防兜底文案漂移）
# ---------------------------------------------------------------------------

def test_all_agent_fallback_texts_covered():
    """全量 agent 兜底文案必须被 token 列表覆盖（防止新增 agent 漏网）。"""
    import re
    from multi_agents import product_agent, tech_agent, billing_agent, complaint_agent, general_agent

    # 抓每个 agent 的 LLM 失败兜底文案（遇到X，请稍后重试 模式）
    fallbacks = []
    for mod in [product_agent, tech_agent, billing_agent, complaint_agent, general_agent]:
        src = open(mod.__file__, encoding="utf-8").read()
        for m in re.finditer(r'response_content = "([^"]*遇到[^"]*请稍后重试[^"]*)"', src):
            fallbacks.append(m.group(1))

    assert len(fallbacks) >= 5, f"应抓到 5 个 agent 兜底文案，实际 {len(fallbacks)}"
    for fb in fallbacks:
        assert _no_system_error(fb) is False, f"兜底文案未覆盖: {fb}"
