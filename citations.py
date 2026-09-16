"""A6/A5 引用溯源：harness 统一附加引用（2026-08-17）。

2026-08-18 拆包重构：从 multi_agent_customer_service.py 逐字迁移，行为零变化。
2026-09-16 多轮适配：指代/省略追问轮并入前序轮引用条目（见 _prior_cited_titles）。

目标：回答标注命中 KB 条目标题，客户/质检可溯源；不碰 agent prompt（agent 只负责答，
引用由 harness 环外加注——引用正确性与内容生成解耦，LLM 无法编造引用）。
口径：与 agent 内部检索共用 kb_retriever.retrieve_titles（同一判定路径：精确→混合→BM25），
保证"标注的就是 agent 用到的"。升级/拒答路径不附加（无检索命中或转人工不适用）。
key 用 create_agent_node 传入的 agent key（product_agent 等），与 graph 节点名一致。
"""

import logging
import os
import re

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
# 脚注解析正则（与 eval_multi_turn._CITATION_RE 同款口径）
_CITATION_RE = re.compile(r"\n\n参考来源[：:]\s*([^\n]+)")
# 指代/省略型追问启发式提示词（宽判定：误报=多引用无害，漏报=真实来源缺失危险）
_ANAPHORA_HINTS = (
    "呢", "那", "不是", "刚才", "上面", "之前", "还没", "它", "这个",
    "那个", "这款", "这台", "他们家", "还要", "再问",
)


def _prior_cited_titles(state: dict, limit: int = 3) -> list:
    """从前序轮 assistant 消息的引用脚注解析沿用条目（最近轮优先，去重保序）。

    多轮适配（2026-09-16，persona-correct 实证）：追问轮回答常沿用前序轮注入
    事实，而本轮 query 单轮化检索与多轮语义脱节——脚注只标本轮检索会完全
    错位（脚注=错发处理而回答=前轮价保）。前序轮脚注是已验证的真实来源
    （harness 自己附加的），环外零成本可得。调用时机在本轮 assistant 消息
    写入 persisted_dialogue 之前，天然只含前序轮。
    """
    out, seen = [], set()
    for msg in reversed(list(state.get("persisted_dialogue") or [])):
        if not isinstance(msg, dict) or msg.get("is_user"):
            continue
        m = _CITATION_RE.search(str(msg.get("content") or ""))
        if not m:
            continue
        for t in re.split(r"[、,，]", m.group(1).strip()):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        if len(out) >= limit:
            break
    return out[:limit]


def _is_followup_query(query: str, state: dict) -> bool:
    """指代/省略型追问启发式：存在前序 assistant 轮 且（query 短 或 含指代提示词）。

    宽判定方向性依据：误报（新话题被误判追问）只是多引用几条前序条目——
    质检代价低；漏报（追问未被识别）则真实来源缺失——脚注失真依旧。
    """
    dialogue = state.get("persisted_dialogue") or []
    has_prior_turn = any(
        isinstance(m, dict) and not m.get("is_user") for m in dialogue
    )
    if not has_prior_turn:
        return False
    return len(query) <= 12 or any(h in query for h in _ANAPHORA_HINTS)


def _prior_cited_context(state: dict, max_chars: int = 4000) -> str:
    """前序轮引用条目全文（检索注入多轮化，2026-09-16 方案 E）。

    追问轮检索 miss 时作为注入回退的事实来源（base_agent._prior_kb_fallback
    调用）：前序轮脚注是 harness 自己附加的已验证真实来源，titles 跨域反查
    全文。无可用返回 ""。与脚注适配（_prior_cited_titles）同源，注入侧与
    标注侧共用同一事实集合——"标注的就是 agent 用到的"契约在追问轮回退
    路径上恢复成立。
    """
    titles = _prior_cited_titles(state)
    if not titles:
        return ""
    try:
        from kb_retriever import get_retriever
        retriever = get_retriever()
        by_title = {}
        for entries in retriever._data.values():
            for e in entries:
                by_title.setdefault(e.get("title", ""), e)
        parts = [f"【{t}】{by_title[t].get('content', '')}"
                 for t in titles if t in by_title]
        return "\n\n".join(parts)[:max_chars]
    except Exception as e:  # noqa: BLE001  回退失败退回升级路径
        logger.warning("前序轮引用条目全文回退失败（跳过）: %s", e)
        return ""


def _attach_citations(agent_name: str, state: dict) -> None:
    """在回答末尾附加命中 KB 条目标题（A6 引用溯源）。无命中/域未知/开关关闭则不改动。

    多轮适配（2026-09-16）：指代/省略追问轮并入前序轮引用条目——本轮检索
    titles 在前（单轮行为前缀零变化），prior 追加在后（去重保序）。
    """
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
        titles = list(_kb_titles(domain, query, top_k=3))
    except Exception as e:  # noqa: BLE001  引用附加失败不影响回答本身
        logger.warning("A6 引用附加失败（跳过）: %s", e)
        return
    # 多轮追问适配：前序轮引用条目并入（本轮在前，prior 追加，去重保序）
    if _is_followup_query(query, state):
        seen = set(titles)
        for t in _prior_cited_titles(state):
            if t not in seen:
                seen.add(t)
                titles.append(t)
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
