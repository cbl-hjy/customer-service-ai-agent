"""图组装：make_graph + 条件路由 + A4/A5 工厂注入。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph

from llm import get_llm
from nodes.state import AgentState
from nodes.classify import classify_query_node
from nodes.compound import compound_node
from nodes.agents import create_agent_node
from nodes.escalate import escalate_node
from nodes.final_response import final_response_node
from trace import _traced
from .checkpointer import get_checkpointer

logger = logging.getLogger(__name__)


def _make_query_rewriter():
    """A4 自适应检索：拒答前 query 改写器（LLM 补全口语/业务术语后重检）。

    触发面：仅混合检索双低拒答路径（95 金标中 ~8 条临界真阳性，如
    "买耳机送什么" dense 0.5097 < 0.55 门禁）。改写语义补全后真阳性分数上推。
    OOD 收口（2026-08-16 实测教训）：盲目改写会把 ood（"快艇怎么修"）业务化为
    "快艇商品售后维修流程"（rerank 0.115/dense 0.609 双过门禁→泄漏），故 prompt
    要求与电商无关的查询输出 OOD → 维持拒答。误判代价对称安全：真阳性误判
    OOD = 维持原拒答（无新增伤害）。
    输出约束：只取首行、≤60 字，防模型输出解释性长文污染检索。
    """
    llm = get_llm()
    sys_prompt = (
        "你是电商客服知识库检索的查询改写器。把口语化、省略、含口头语的用户查询"
        "改写成适合知识库检索的完整查询：\n"
        "- 补全省略的业务名词（如\"拍一个\"→\"拍下订单一件\"）\n"
        "- 把口头语替换为业务术语（\"送东西\"→\"赠品 满赠活动\"，"
        "\"订了几个\"→\"订单商品数量查询\"）\n"
        "- 严格保持原意，不添加原查询没有的诉求\n"
        "- 查询已完整清晰则原样返回\n"
        "**OOD 判定（最高优先级）：如果查询与电商购物/订单/售后/产品/物流/会员"
        "完全无关（如交通工具维修、乐器演奏、天文地理等），只输出 OOD 三个字母，"
        "不要改写**——把它强行改写成业务查询会造成错误回答。\n"
        "只输出改写后的查询文本或 OOD，不要任何解释。"
    )

    def rewrite(domain: str, query: str):
        try:
            r = llm.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=f"业务域：{domain}\n用户查询：{query}"),
            ])
            first = str(r.content or "").strip().split("\n")[0].strip()
            if first.upper().startswith("OOD") or first.upper() in ("O", "OD"):
                return None  # OOD → 维持拒答（改写器判定与业务无关）
            return first[:60] or None
        except Exception as e:  # noqa: BLE001  改写失败 → 维持拒答主路径
            logger.warning("A4 query 改写调用失败（维持拒答）: %s", e)
            return None

    return rewrite


def _make_query_decomposer():
    """A5 复合查询：拆解器工厂（LLM 把复合查询拆为独立子查询，各带业务域）。

    仅在 classify 阶段 detect_compound 命中多诉求后才调用（零成本前置）。
    fail-safe：通道故障/异常由 _try_decompose 捕获回退单路径，不阻断主流程。
    输出为 JSON 数组字符串，由 tools.query_decompose.parse_decomposed 解析规范化。
    """
    llm = get_llm()
    from tools.query_decompose import _DECOMPOSE_SYSTEM_PROMPT

    def decompose(query: str):
        from langchain_core.messages import HumanMessage, SystemMessage
        r = llm.invoke(
            [SystemMessage(content=_DECOMPOSE_SYSTEM_PROMPT), HumanMessage(content=f"请拆解以下查询：{query}")],
            response_format={"type": "json_object"},
        )
        return str(getattr(r, "content", "") or "").strip()

    return decompose


def make_graph():
    """构建LangGraph工作流图"""
    # A4 自适应检索（2026-08-16）：注入 query 改写器。评估脚本不 import 本模块
    # 即不注入，保持纯检索口径；注入失败不影响主流程（拒答路径行为退回原样）。
    try:
        import kb_retriever as _kbr
        _kbr.set_query_rewriter(_make_query_rewriter())
    except Exception as e:  # noqa: BLE001
        logger.warning("A4 query 改写器注入失败（自适应检索退化为直接拒答）: %s", e)

    # A5 复合查询（2026-08-17）：注入查询拆解器（与 A4 同纪律——make_graph 注入，
    # 评估纯检索口径/单测不注入则 _try_decompose 返回 [] 走原路径）。
    try:
        import tools.query_decompose as _qdec
        _qdec.set_query_decomposer(_make_query_decomposer())
    except Exception as e:  # noqa: BLE001
        logger.warning("A5 查询拆解器注入失败（复合查询退化为单路径）: %s", e)

    # 创建工作流图
    workflow = StateGraph(AgentState)

    # 添加节点（V18：全部包 _traced 节点包装器——全链路追踪，零侵入业务代码）
    workflow.add_node("classify_query", _traced("classify_query", classify_query_node))
    workflow.add_node("product_agent", _traced("product_agent", create_agent_node("product_agent")))
    workflow.add_node("tech_agent", _traced("tech_agent", create_agent_node("tech_agent")))
    workflow.add_node("billing_agent", _traced("billing_agent", create_agent_node("billing_agent")))
    workflow.add_node("complaint_agent", _traced("complaint_agent", create_agent_node("complaint_agent")))
    workflow.add_node("general_agent", _traced("general_agent", create_agent_node("general_agent")))
    workflow.add_node("compound", _traced("compound", compound_node))
    workflow.add_node("escalate", _traced("escalate", escalate_node))
    workflow.add_node("final_response", _traced("final_response", final_response_node))

    # 设置入口点
    workflow.set_entry_point("classify_query")

    # 条件路由（C4）：升级决策（代码硬边界）→ escalate；否则按分类路由业务 agent
    def route_after_classify(state):
        # 【韧性】LLM 通道故障（403/熔断/5xx）降级提示：直接给友好提示，不再进业务 agent
        # （业务 agent 会再次调 LLM，已熔断会覆盖提示且徒增无效请求）
        if state.get("llm_unavailable"):
            return "final_response"
        if state.get("query_type") == "out_of_scope":
            return "out_of_scope"  # 护栏：直接拒绝（classify 节点已设固定回复）
        # A5 复合查询（2026-08-17）：多诉求拆解命中 → 优先走综合回答节点
        # （classify 已强制 escalate=False；即使残留 escalate 标记也以复合为准）
        if state.get("compound_subs"):
            return "compound"
        if state.get("escalated"):
            return "escalate"
        return state.get("query_type", "general_inquiry")

    workflow.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "product_info": "product_agent",
            "technical_support": "tech_agent",
            "billing": "billing_agent",
            "complaint": "complaint_agent",
            "general_inquiry": "general_agent",
            "out_of_scope": "final_response",
            "compound": "compound",
            "escalate": "escalate",
            "final_response": "final_response",
        }
    )

    # 添加直接边（所有节点都连接到最终响应）
    workflow.add_edge("product_agent", "final_response")
    workflow.add_edge("tech_agent", "final_response")
    workflow.add_edge("billing_agent", "final_response")
    workflow.add_edge("complaint_agent", "final_response")
    workflow.add_edge("general_agent", "final_response")
    workflow.add_edge("compound", "final_response")
    workflow.add_edge("escalate", "final_response")

    # 设置结束点
    workflow.set_finish_point("final_response")

    # 编译工作流（checkpointer：SqliteSaver 文件持久化，跨进程/重启保留 persisted_dialogue）
    app = workflow.compile(checkpointer=get_checkpointer())

    logger.info("LangGraph 工作流图构建完成")
    return app
