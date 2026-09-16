"""集成测试：demo 工单集 8 条全流程（需要 API key，无 key 自动跳过）

断言策略（LLM 分类有随机性，不 assert 死 label）：
- 粗粒度：升级/不升级的决策符合业务预期（硬边界，稳定）
- label 允许合理集合（如退款可 billing/general_inquiry）

⚠️ 隔离：本测试真实 invoke 完整图 + checkpointer（有 key 时），
必须用独立 CHECKPOINT_DB_PATH / ARCHIVE_DB_PATH（fixture 隔离），
否则每次回归都写入生产 checkpoints.db / dialogue_archive.db（it-guardrail 52 轮污染的教训，2026-08-14）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from config import OPENAI_API_KEY

pytestmark = [
    pytest.mark.skipif(not OPENAI_API_KEY, reason="无 API key，跳过集成测试"),
    pytest.mark.api,  # 真实 LLM API（fast 门禁排除）
]


@pytest.fixture(scope="session", autouse=True)
def _isolate_dbs(tmp_path_factory):
    """整个测试会话用独立 DB（checkpoints + archive + tickets），防污染生产库。

    用 session 级：_app / checkpointer 是模块级单例（进程内复用），
    per-test 换路径会让单例绑定到第一个路径而失效。
    """
    base = tmp_path_factory.mktemp("it-dbs")
    os.environ["CHECKPOINT_DB_PATH"] = str(base / "cp.db")
    os.environ["ARCHIVE_DB_PATH"] = str(base / "archive.db")
    os.environ["TICKETS_DB_PATH"] = str(base / "tickets.db")
    yield


from multi_agent_customer_service import make_graph

_app = None


def _get_app():
    global _app
    if _app is None:
        _app = make_graph()
    return _app


# (工单, 期望是否升级, label 合理集合)
TICKETS = [
    ("你们最新款手机 X3 Lite 多少钱？和 X2 Max 比哪个性价比高？", False, {"product_info"}),
    ("我的手机升级系统后一直闪退，重启也没用", False, {"technical_support"}),
    ("昨天买的耳机想退掉，7 天无理由怎么申请？", False, {"billing", "general_inquiry"}),
    ("物流太慢了，等了一周还没到，我要投诉你们！", True, {"complaint"}),
    ("你们客服几点上班？电话多少？", False, {"general_inquiry"}),
    ("帮我查一下订单 20260812 的物流进度", False, {"general_inquiry"}),
    ("帮我写一首关于夏天的诗", False, {"out_of_scope"}),  # 护栏：不升级但拒绝
    ("退款 10 天没到账，中间催了 3 次没人理，我要找你们经理", True, {"billing", "complaint", "general_inquiry"}),
    # v15 修复：用户主动求人工 → 强制升级（代码硬边界，即使分类为 general_inquiry）
    ("怎么联系人工客服？有严重问题当面说", True, {"general_inquiry", "complaint", "billing"}),
]


@pytest.mark.parametrize("query, expect_escalate, allowed_labels", TICKETS, ids=[f"t{i+1}" for i in range(len(TICKETS))])
def test_ticket_flow(query, expect_escalate, allowed_labels):
    app = _get_app()
    result = app.invoke(
        {"customer_query": query},
        {"configurable": {"thread_id": f"it-{abs(hash(query)) % 10000}"}},
    )
    # 1) 分类 label 在合理集合（容忍 LLM 随机性）
    assert result["query_type"] in allowed_labels, (
        f"label={result['query_type']} 不在预期集合 {allowed_labels}"
    )
    # 2) 升级决策符合硬边界预期
    if expect_escalate:
        assert result["escalated"] is True, f"应升级但未升级: {result['escalation_reason']}"
        assert result["escalation_reason"], "升级必须有原因"
    else:
        if result["query_type"] != "out_of_scope":
            assert result["escalated"] is False, f"不应升级却升级: {result['escalation_reason']}"
    # 3) 回复非空
    assert result["response"], "回复为空"


def test_out_of_scope_guardrail_not_weakened():
    """护栏不被升级机制削弱：out_of_scope 走固定拒绝，不升级、不硬答"""
    app = _get_app()
    result = app.invoke(
        {"customer_query": "帮我写一首关于夏天的诗"},
        {"configurable": {"thread_id": "it-guardrail"}},
    )
    assert result["query_type"] == "out_of_scope"
    assert result["escalated"] is False
    assert "抱歉" in result["response"] or "客服" in result["response"]
