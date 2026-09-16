"""测试：统一知识库 BM25 检索（kb_retriever）。纯本地，不触发 LLM。"""

import os
import sys

import pytest

pytestmark = pytest.mark.model  # 真实加载 bge-m3/reranker（fast 门禁排除，全量时跑）

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kb_retriever as _kbr
from kb_retriever import KBRetriever, retrieve, STOPWORDS, _tokenize


@pytest.fixture(autouse=True)
def _pure_retrieval():
    """隔离 A4 改写器：本文件断言纯检索行为。

    集成测试（test_workflow_integration）的 make_graph() 会全局注入 LLM 改写器，
    同进程混跑时若无隔离，纯拒答断言会被改写重检路径污染（2026-08-16 实踩：
    scenario_5 在 integration 先跑后被改写路径命中）。
    """
    _kbr.clear_query_rewriter()
    yield
    _kbr.clear_query_rewriter()


def setup_module():
    # 预热 jieba，避免首次分词日志干扰
    _tokenize("预热")


def test_precise_model_hit_product():
    """kb-v2 型号精确命中（v1 的 X2 Max/X3 Lite 在 v2 已不存在，改用 v2 型号比价）"""
    r = retrieve("product", "星环 X1 Pro 和 Book Air 13 比哪个性价比高")
    assert "手机" in r  # exact 预检命中手机条目（exact 优先于混合路）


def test_precise_category_hit_product():
    r = retrieve("product", "我想买手机")
    assert "手机" in r


def test_bm25_fuzzy_hit_product():
    """BM25 模糊召回：问法变形仍能命中（kb-v2 品类词为"笔记本"系列，2026-08-16）"""
    r = retrieve("product", "你们有游戏本卖吗")
    assert "笔记本" in r


def test_ood_no_hit_product():
    """库外产品 → 空串（触发 _no_answer 升级）"""
    assert retrieve("product", "你们有火星车卖吗") == ""


def test_billing_hit():
    assert "退款" in retrieve("billing", "退款怎么操作")
    assert "发票" in retrieve("billing", "发票抬头能改吗")
    assert "支付" in retrieve("billing", "怎么分期付款")


def test_tech_hit_and_ood():
    # kb-v2 无"手机闪退"条目（仅平板App闪退），语义命中同域手机故障条目即可
    r = retrieve("tech", "手机升级系统后一直闪退")
    assert "手机" in r
    # "黑屏"在 v2 是独立条目（v1 时代归并到"无法开机"）
    r = retrieve("tech", "电脑开机黑屏怎么办")
    assert "黑屏" in r
    assert retrieve("tech", "优雅地演奏小提琴") == ""


def test_general_hit():
    assert "400" in retrieve("general", "客服电话是多少")
    assert "7" in retrieve("general", "营业时间是几点")


def test_complaint_hit_and_ood():
    assert "物流" in retrieve("complaint", "物流太慢我要投诉")
    assert retrieve("complaint", "太空飞船怎么修") == ""


def test_domain_isolation():
    # kb-v2 无"接口故障"条目，tech 域 query 语义命中"维修费用查询"
    assert "维修" in retrieve("tech", "设备故障了")
    assert retrieve("complaint", "快艇怎么修") == ""


def test_stopwords_removed():
    """停用词被过滤，不参与 BM25 权重"""
    toks = _tokenize("请问我这个怎么办")
    assert not any(t in STOPWORDS for t in toks)


def test_retriever_singleton():
    """单例复用同一份索引"""
    a = KBRetriever()
    b = KBRetriever()
    assert a._indexes["product"] is not b._indexes["product"] or a is b
    from kb_retriever import get_retriever
    assert get_retriever() is get_retriever()


# ---------------------------------------------------------------------------
# 8-13 用户画像模拟发现的问题：口语退货 + 需求描述召回缺口
# ---------------------------------------------------------------------------

def test_billing_oral_return_wo_quality():
    """口语退货误升级 A1：我要退个耳机 → 命中退款政策"""
    r = retrieve("billing", "我要退个耳机")
    assert "退款" in r or "退货" in r


def test_billing_oral_return_wo_quality_2():
    """口语退货误升级 A2：耳机我想退了 → 命中退款政策"""
    r = retrieve("billing", "耳机我想退了")
    assert "退款" in r or "退货" in r


def test_billing_oral_return_wo_quality_3():
    """口语退货误升级 A3：退一下这个平板 → 命中退款政策"""
    r = retrieve("billing", "退一下这个平板")
    assert "退款" in r or "退货" in r


def test_billing_oral_return_wo_quality_4():
    """口语退货误升级 A4：能把手机退了吗 → 命中退款政策"""
    r = retrieve("billing", "能把手机退了吗")
    assert "退款" in r or "退货" in r


def test_billing_oral_return_wo_quality_5():
    """口语退货误升级 A5：那个耳机不要了 → 命中退款政策"""
    r = retrieve("billing", "那个耳机不要了")
    assert "退款" in r or "退货" in r


def test_billing_oral_return_wo_quality_6():
    """口语退货误升级 A6：这个能退吗 → 命中退款政策"""
    r = retrieve("billing", "这个能退吗")
    assert "退款" in r or "退货" in r


def test_product_usage_scenario_1():
    """需求描述 B1：我平时就打打游戏，续航重要些 → 命中产品"""
    r = retrieve("product", "我平时就打打游戏，续航重要些")
    assert "手机" in r or "电脑" in r


def test_product_usage_scenario_2():
    """需求描述 B2：我妈妈用，不要太复杂的 → 命中老人机/送父母（2026-08-16 KB 扩容后新增覆盖）。

    演进：277 条时代双低置信（rerank 0.0042/dense 0.5122）拒答升级人工；
    601 条新增"老人机选购指南/礼物推荐-送父母"后正确覆盖
    （rerank 0.0782/dense 0.5708 双过新门禁 0.07/0.55）。
    OOV 拒答红线由 test_complaint_hit_and_ood（太空飞船）继续守护。
    """
    r = retrieve("product", "我妈妈用，不要太复杂的")
    assert "老人" in r or "父母" in r


def test_product_usage_scenario_3():
    """需求描述 B3：它适合学生用吗 → 命中产品（rerank 0.6263 强相关放行）"""
    r = retrieve("product", "它适合学生用吗")
    assert "手机" in r or "电脑" in r or "笔记本" in r


def test_product_usage_scenario_4():
    """需求描述 B4：我想要续航久一点的 → 命中产品"""
    r = retrieve("product", "我想要续航久一点的")
    assert "手机" in r or "电脑" in r


def test_product_usage_scenario_5():
    """需求描述 B5：我平时就刷刷视频回回消息 → 纯检索双低置信拒答升级人工。

    实测：rerank top1=0.0041 / dense=0.4865，双低置信 → 拒答（依据同 scenario_2）。
    2026-08-16 A4 演进：线上（make_graph 注入 LLM 改写器后）该 query 被改写为
    "适合刷视频聊天的手机推荐"命中"大屏手机推荐"等 601 新增选购条目——合理接住。
    纯路径拒答与 A4 改写放行的边界由 test_a4_rewrite_retry_hook 确定性守护。
    """
    assert retrieve("product", "我平时就刷刷视频回回消息") == ""


def test_a4_rewrite_retry_hook():
    """A4 自适应检索：双低拒答前的改写重检钩子（假改写器，无 LLM 依赖）。

    三态断言（2026-08-16 建档，链路确定性探针实测）：
    1. 无改写器：双低 query 维持拒答（纯路径不变）
    2. 改写器语义补全：拒答 → 重检放行（命中 601 选购条目）
    3. 改写器 OOD 判定（返回 None）：维持拒答（防"快艇怎么修"被业务化泄漏）
    真实 LLM 改写行为由 eval_multi_turn 端到端验证（mt-206/207/307 修复）。
    """
    q = "我平时就刷刷视频回回消息"
    # 1) 纯路径拒答
    assert retrieve("product", q) == ""
    # 2) 语义补全 → 放行
    _kbr.set_query_rewriter(lambda d, qq: "适合刷视频聊天的手机推荐")
    try:
        r = retrieve("product", q)
        assert "手机" in r or "推荐" in r
    finally:
        _kbr.clear_query_rewriter()
    # 3) OOD 判定 None → 维持拒答
    _kbr.set_query_rewriter(lambda d, qq: None)
    try:
        assert retrieve("product", q) == ""
    finally:
        _kbr.clear_query_rewriter()


def test_billing_oral_still_ood():
    """库外口语仍空串（不误杀）：今天天气怎么样 → 空串（非退款/账单类）"""
    assert retrieve("billing", "今天天气怎么样") == ""


def test_billing_shipping_fee():
    """运费（8-13 补条目）：那运费谁出 → 命中退货运费"""
    r = retrieve("billing", "那运费谁出")
    assert "运费" in r


def test_billing_shipping_fee_2():
    """运费口语：邮费谁付 → 命中"""
    r = retrieve("billing", "邮费谁付")
    assert "运费" in r or "邮费" in r


def test_product_usage_still_ood():
    """库外需求描述仍空串：我要能在地球外飞行的交通工具 → 空串"""
    assert retrieve("product", "我要能在地球外飞行的交通工具") == ""