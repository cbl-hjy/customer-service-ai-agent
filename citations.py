"""A6/A5 引用溯源：harness 统一附加引用（2026-08-17）。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。

目标：回答标注命中 KB 条目标题，客户/质检可溯源；不碰 agent prompt（agent 只负责答，
引用由 harness 环外加注——引用正确性与内容生成解耦，LLM 无法编造引用）。
口径：与 agent 内部检索共用 kb_retriever.retrieve_titles（同一判定路径：精确→混合→BM25），
保证"标注的就是 agent 用到的"。升级/拒答路径不附加（无检索命中或转人工不适用）。
key 用 create_agent_node 传入的 agent key（product_agent 等），与 graph 节点名一致。
"""

import logging
import os

logger = logging.getLogger(__name__)

_AGENT_DOMAIN_MAP = {
    "product_agent": "product",
    "tech_agent": "tech",
    "billing_agent": "billing",
    "complaint_agent": "complaint",
    "general_agent": "general",
}
# 引用开关：默认开（产品行为）；评估比对/调试可经环境变量关闭（加法，不改变既有路径）
ENABLE_CITATIONS = os.environ.get("ENABLE_CITATIONS", "1") == "1"
# 引用脚注格式（纯文本，避免【】与内部泄漏正则碰撞；客户可见，前缀固定便于 eval 解析）
_CITATION_FOOTER_PREFIX = "\n\n参考来源："


def _attach_citations(agent_name: str, state: dict) -> None:
    """在回答末尾附加命中 KB 条目标题（A6 引用溯源）。无命中/域未知/开关关闭则不改动。"""
    if not ENABLE_CITATIONS:
        return
    domain = _AGENT_DOMAIN_MAP.get(agent_name)
    if not domain:
        return
    query = state.get("customer_query", "")
    if not query:
        return
    try:
        from kb_retriever import retrieve_titles as _kb_titles
        titles = _kb_titles(domain, query, top_k=3)
    except Exception as e:  # noqa: BLE001  引用附加失败不影响回答本身
        logger.warning("A6 引用附加失败（跳过）: %s", e)
        return
    if not titles:
        return
    resp = str(state.get("response") or "")
    if not resp:
        return
    cited = "、".join(t for t in titles if t)
    if not cited:
        return
    state["response"] = resp + f"{_CITATION_FOOTER_PREFIX}{cited}"
    state.setdefault("tools_used", []).append("citation")


def _attach_compound_citations(state: dict, hits: list) -> None:
    """A5 复合回答引用溯源：合并各子查询命中标题（从检索文本解析，与注入内容一致），
    附加"参考来源"脚注（幂等，去重保序）。judge 核对基座按引用 titles 取条目全文，
    因此引用必须覆盖实际注入的 KB 条目，否则 judge 缺依据误判。"""
    if not ENABLE_CITATIONS:
        return
    seen, titles = set(), []
    for _sub, _domain, _text, kb_titles in hits:
        for t in kb_titles:
            if t and t not in seen:
                seen.add(t)
                titles.append(t)
    if not titles:
        return
    resp = str(state.get("response") or "")
    if not resp:
        return
    state["response"] = resp + f"{_CITATION_FOOTER_PREFIX}{'、'.join(titles)}"
    state.setdefault("tools_used", []).append("citation")
