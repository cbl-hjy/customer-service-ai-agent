"""V15 测试：死代码清理（product_database 大字典 / AgentState.memory / 死导入）。

背景（审查清单 V15）：product_agent.product_database 自检索层收敛（kb_retriever）后
无人读取；AgentState.memory 初始化 None 后全项目零写入；BaseChatMessageHistory 导入
随 memory 删除成死导入。LOG_CONFIG 经核查非死代码（V9 已由 setup_logging() 使用）。

验证方式：结构性断言（死代码不存在）+ 行为回归（删除后功能不受影响）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _src_of(module_path: str) -> str:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), module_path)
    with open(p, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 结构断言：死代码已删除
# ---------------------------------------------------------------------------

def test_product_database_removed():
    """product_database 大字典 + TODO 注释已删除（断言代码形态，注释可提及历史）。"""
    src = _src_of("multi_agents/product_agent.py")
    assert "self.product_database = {" not in src, "product_database 大字典残留"
    assert '"推荐指数"' not in src, "模拟数据残留（大字典里的字段）"
    assert "TODO: 产品信息应该从数据库获取" not in src, "遗留 TODO 未清"


def test_product_match_still_uses_kb():
    """_match_products 仍走统一知识库（删除字典不影响检索路径）。"""
    src = _src_of("multi_agents/product_agent.py")
    assert 'kb_retrieve("product", query)' in src, "检索路径被破坏"


def test_agent_state_memory_removed():
    """AgentState.memory 字段声明 + classify 初始化已删除。"""
    src = _src_of("multi_agent_customer_service.py")
    assert '"memory"' not in src, "memory 字段残留"
    assert "state[\"memory\"]" not in src, "memory 初始化残留"


def test_base_chat_message_history_import_removed():
    """BaseChatMessageHistory 死导入已删除。"""
    src = _src_of("multi_agent_customer_service.py")
    assert "BaseChatMessageHistory" not in src, "死导入残留"


def test_log_config_is_alive():
    """LOG_CONFIG 不是死代码（V9 起由 setup_logging 使用，V15 保留）。"""
    src = _src_of("config.py")
    assert "LOG_CONFIG" in src
    assert "setup_logging()" in src, "setup_logging 调用点被误删"


# ---------------------------------------------------------------------------
# 行为回归：删除后功能正常
# ---------------------------------------------------------------------------

def test_product_agent_initializes_and_matches():
    """ProductAgent 初始化 + 检索路径正常（无 product_database 依赖）。"""
    from multi_agents.product_agent import ProductAgent
    agent = ProductAgent()
    # 无 product_database 属性（已删）
    assert not hasattr(agent, "product_database")
    # 检索路径可用（无 key 环境走 kb_retrieve，纯本地 BM25，不依赖 LLM）
    # kb-v2：X3 Lite 不存在，语义命中"星环 X1 Lite 入门手机规格"（X 系列 Lite 版）
    result = agent._match_products("X3 Lite 手机多少钱")
    assert isinstance(result, str)
    assert "X1 Lite" in result or "手机" in result, f"检索结果异常: {result[:50]}"


def test_graph_builds_without_memory_field():
    """图构建正常（AgentState 无 memory 字段不影响 checkpointer 序列化）。"""
    from multi_agent_customer_service import AgentState, make_graph
    # TypedDict 无 memory 键
    assert "memory" not in AgentState.__annotations__
    # 图构建不抛异常（无 key 环境也能构建，FakeLLM 不需要）
    import os as _os
    _os.environ.setdefault("CHECKPOINT_DB_PATH", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "checkpoints.db"))
    app = make_graph()
    assert app is not None
