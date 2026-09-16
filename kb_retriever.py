#!/usr/bin/env python3
"""
统一知识库检索模块（BM25，kb-v1）。

背景：原 5 个 agent 各自持有 *_database + 硬编码 _match_* 关键词匹配，逻辑重复且
"关键词兜底过宽"（如"支付"命中"支付方式"）。本模块把知识获取收敛为单一检索层，
各 agent 只按 domain 调用，输出格式与原有 _match_* 兼容（product 用"产品：…"，
其余域用"【category】\n• title：content"），无命中返回空串 → 触发既有 _no_answer 升级。

检索双路（精确优先，模糊兜底）：
  1. 精确预检：query 含条目标题 / 型号 / 检索词 → 直接命中（保住 v02 型号匹配回归）
  2. BM25 模糊召回：jieba 分词 + BM25Okapi，按分数取 Top-K，低于阈值返回空
域隔离：每个 domain 独立建索引，避免跨域串扰。

红线：技术按需叠加——BM25+jieba 为基座，2026-08-16 经 95 条金标评估数据支撑后
      接入混合检索（bge-m3 稠密 + RRF 融合 + bge-reranker-v2-m3 精排，R@1 0.374→0.468，
      hard/ambiguous 档 +39%/+75%），ENABLE_HYBRID 开关可一键回滚纯 BM25；
      2026-09-16 融合参数定版 k=20/w_dense=1.5（217 金标：加权 RRF R@1 0.493 /
      +rerank 0.482，回滚 = hybrid_retriever.RRF_K=60/RRF_W_DENSE=1.0，详见
      ai-docs/w5-rag-batch1.md）；无命中不硬答，返回空串由 harness 走升级（C5 不变）。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Union

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 知识库：kb-v2（277 条，5 域，由 eval/kb_v2/generate_kb.py 生成，2026-08-16 上线）。
# v1（data/knowledge_base.json，62 条）保留作回滚对照。
_KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base_v2.json")

# 结果里展示的字段（title/content/category），检索词与型号不展示
_DISPLAY_FIELDS = ("title", "content", "category")

# 无命中阈值：BM25 分数 < 该值视为无相关（避免噪声召回）
SCORE_THRESHOLD = 0.30

# 每个 domain 召回条数
TOP_K = 3

# 混合检索开关（P1 RRF + P2 reranker，2026-08-16 落地，95 条金标 R@1 0.374→0.468）。
# False = 回滚纯 BM25 行为（精确预检 + BM25 top-3）。
ENABLE_HYBRID = True

# 混合路的 BM25 召回宽度：RRF 融合需要宽池（原 TOP_K=3 太窄），与评估脚本同口径 20。
HYBRID_BM25_TOP_K = 20

# 混合路应答门禁（2026-08-16，95 条金标 + 6 条 ood 实测校准）：
#   rerank top1 >= RERANK_CONF_FLOOR   → 答（reranker 强信号）
#   dense top1 >= DENSE_OOD_FLOOR      → 答（稠密强信号；BM25 空召回时也靠它兜底）
#   两者都低 → 拒答（返回空串 → 升级人工）。数据：ood 6 条全 <(0.02, 0.55)，
#   金标误伤 6/95（rerank 本就低置信，拒答转人工优于错答）。
# 2026-08-16 标定（KB 277→601 扩容后重标定，95 金标仿真）：
#   0.02→0.07：金标放行 89/95 不变（0.02-0.07 区间真阳性为空集），
#   同时挡住"快艇怎么修"假阳性（rerank=0.0612 语义漂移命中质量问题补偿标准）。
#   扩容后域内条目变多，rerank 噪声分普遍抬升，旧 0.02 门禁余量被挤掉。
RERANK_CONF_FLOOR = 0.07
DENSE_OOD_FLOOR = 0.55

# 无区分度停用词：疑问词/语气词/通用动词，剔除避免 BM25 虚高 IDF 误召回 OOV 查询
# （如"太空飞船怎么修"里的"怎么"在 complaint 域高频，会把无关条目推过阈值）
STOPWORDS = {
    "怎么", "怎么办", "如何", "怎样", "请问", "我想", "我要", "你们", "你好",
    "您好", "个", "了", "吗", "呢", "啊", "吧", "的", "是", "在", "有", "我",
    "你", "他", "她", "它", "这", "那", "什么", "哪些", "那个", "这个", "下",
    "一下", "可以", "能", "吗",
    # 2026-08-16 kb-v2 上线时补：单字动作词/模板词，IDF 虚高污染 BM25
    #   "修"：complaint 域 keywords 含"反复维修/修了又坏"，ood query"太空飞船怎么修"被单字推过阈值
    #   "操作"：v2 模板生成 content 通用词，"退款怎么操作"被"会员价规则"等含"操作"条目压过退款条目
    "修", "操作",
}

# =============================================================================
# 查询扩展层（2026-08-13，方向：治"词不匹配类"检索漏，不碰 agent 决策）
# 背景：泛化缺口量化为"新问法/新实体词漏"。查询扩展把用户 Query 里的同义/上下位词
#       追加出知识库词，让精确预检与 BM25 都能命中"叫法不同但同一物"的条目。
# 原则：
#   - 只追加不替换：扩展词 append 到 query 后，保留原词语义（避免丢失原意图）。
#   - 集中式一张映射表：比"往 knowledge_base 逐个塞词"更可维护，一处维护多域受益。
#   - 救"词不匹配类"漏；救不了"知识库根本没有条目"的漏（那是补知识域的事）。
# =============================================================================
# 同义/上下位映射：{用户词: [知识库词/候选扩展]}
_QUERY_EXPANSION = {
    # 配件/随箱件（用户叫法 → 知识库/通用词）
    "电源适配器": ["充电器", "充电头", "电源"],
    "适配器": ["充电器", "充电头", "电源"],
    "笔尖": ["笔头", "替换笔尖", "配件"],
    "数据线": ["充电线", "USB线", "充电线"],
    "充电线": ["数据线", "USB线"],
    "保护壳": ["外壳", "手机壳", "保护套"],
    "贴膜": ["屏幕膜", "钢化膜", "保护膜"],
    "支架": ["充电支架", "手机支架"],
    "挂绳": ["挂带", "腕带"],
    # 售后/保修叫法（归一）
    "质保": ["保修", "质保期", "保修期"],
    "保修": ["质保", "保修期"],
    "免费修": ["保修", "保修期内免费"],
    "寄回来修": ["寄修", "返修", "保修"],
    "补寄": ["补发", "重发", "漏发"],
    "少寄": ["少件", "漏发", "缺配件"],
    "换一根": ["换新", "更换", "换货"],
    # 物流/时效叫法
    "快递单号": ["物流单号", "快递单号查询"],
    "派送": ["配送", "送货", "派件"],
    "今天到": ["今日送达", "时效", "配送"],
    # 满赠/赠品口语叫法（C 实验召回缺口 4 条：时效问赠品的口语词不匹配，2026-09-16）
    "送什么": ["满赠", "赠品"],
    "送啥": ["满赠", "赠品"],
    "送一盒": ["满赠", "赠品"],
    "拍了很多": ["满赠", "赠品"],
    # 选型/品类
    "头戴": ["头戴式", "耳机"],
    "入耳": ["入耳式", "耳机"],
    "无线": ["蓝牙", "无线蓝牙"],
}


def _expand_query(query: str) -> str:
    """查询扩展：把 query 中命中的映射词追加扩展词，返回扩展后 query（只增不替换）。

    示例：query="电源适配器能换吗" → "电源适配器能换吗 充电器 充电头 电源"
    扩展词只追加，不丢失原词；未命中映射则原样返回，零开销。
    """
    if not query:
        return query
    lower = query.lower()
    extra: List[str] = []
    for kw, exps in _QUERY_EXPANSION.items():
        if kw in lower:
            for exp in exps:
                if exp not in query and exp not in extra:
                    extra.append(exp)
    if not extra:
        return query
    return query + " " + " ".join(extra)


def _tokenize(text: str) -> List[str]:
    """jieba 精确模式分词，过滤停用词与空白（保留英文型号/数字）。"""
    toks = jieba.lcut(text.lower())
    out = []
    for t in toks:
        t = t.strip()
        if not t or t in STOPWORDS:
            continue
        out.append(t)
    return out


class KBRetriever:
    """统一知识库检索器：按 domain 构建 BM25 索引，双路检索。"""

    def __init__(self, kb_path: Optional[str] = None) -> None:
        self._kb_path = kb_path or _KB_PATH
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._indexes: Dict[str, BM25Okapi] = {}
        self._tokenized_corpus: Dict[str, List[List[str]]] = {}
        self._load()

    def _load(self) -> None:
        with open(self._kb_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for domain, entries in raw.items():
            if domain.startswith("_"):
                continue
            self._data[domain] = entries
        self._build_indexes()

    def _tokenize(self, text: str) -> List[str]:
        """jieba 精确模式分词，过滤停用词与空白（保留英文型号/数字）。"""
        return _tokenize(text)

    def _entry_search_text(self, entry: Dict[str, Any]) -> str:
        """条目的检索文本 = 展示字段 + 检索词 + 型号（用于 BM25 召回）。"""
        parts = [
            entry.get("category", ""),
            entry.get("title", ""),
            entry.get("content", ""),
        ]
        kws = entry.get("keywords", [])
        if isinstance(kws, list):
            parts.extend(str(k) for k in kws)
        models = entry.get("models", [])
        if isinstance(models, list):
            parts.extend(str(m) for m in models)
        return " ".join(p for p in parts if p)

    def _build_indexes(self) -> None:
        for domain, entries in self._data.items():
            tokenized = [self._tokenize(self._entry_search_text(e)) for e in entries]
            self._tokenized_corpus[domain] = tokenized
            self._indexes[domain] = BM25Okapi(tokenized)

    def _exact_hits(self, domain: str, query: str) -> List[Dict[str, Any]]:
        """精确预检：query 含条目标题 / 型号 / 单个检索词 → 命中。"""
        q = query.lower()
        hits: List[Dict[str, Any]] = []
        for entry in self._data.get(domain, []):
            title = str(entry.get("title", "")).lower()
            if title and title in q:
                hits.append(entry)
                continue
            models = entry.get("models", [])
            if isinstance(models, list) and any(str(m).lower() in q for m in models if m):
                if entry not in hits:
                    hits.append(entry)

            # 检索词：只做"整词包含"，避免"支付"命中"支付方式"也误伤——但与原本地兜底一致即可
            kws = entry.get("keywords", [])
            if isinstance(kws, list):
                for k in kws:
                    if k and str(k).lower() in q:
                        if entry not in hits:
                            hits.append(entry)
                        break
        return hits

    def _bm25_hits(self, domain: str, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """BM25 模糊召回：分数 >= 阈值的 Top-K。"""
        index = self._indexes.get(domain)
        if index is None:
            return []
        q_toks = self._tokenize(query)
        if not q_toks:
            return []
        scores = index.get_scores(q_toks)
        entries = self._data.get(domain, [])
        ranked = sorted(
            ((scores[i], entries[i]) for i in range(len(entries))),
            key=lambda x: x[0],
            reverse=True,
        )
        hits = []
        for score, entry in ranked:
            if score < SCORE_THRESHOLD:
                break
            hits.append(entry)
            if len(hits) >= top_k:
                break
        return hits

    def _format_result(self, domain: str, entries: List[Dict[str, Any]]) -> str:
        """按 agent 原有 _match_* 的输出格式拼装。product 用"产品：…"，其余用"【category】…"。

        A1 扩容后（601 条）product 条目 content 为段落式（【推荐方案】…），无"品牌："等
        字段前缀行；旧字段解析（_product_*）会对段落式条目返回全空，导致注入内容与
        judge 基座缺失、agent 只能靠标题+常识自由发挥（幻觉温床，mt-205 实测）。修复：
        字段式条目走字段解析（历史格式兼容），段落式条目直接输出完整 content。
        """
        if not entries:
            return ""
        if domain == "product":
            blocks = []
            for e in entries:
                content = str(e.get("content", ""))
                title = e.get("title", "")
                if content.strip().startswith("品牌："):
                    blocks.append(
                        f"""产品：{title}
品牌：{self._product_brand(e)}
型号：{self._product_models(e)}
价格区间：{self._product_price(e)}
主要特点：{self._product_features(e)}
适用人群：{self._product_users(e)}
推荐指数：{self._product_rating(e)}"""
                    )
                else:
                    blocks.append(f"产品：{title}\n{content.strip()}")
            return "\n\n".join(blocks)

        # 其余域：按 category 分组，每组输出全部命中条目
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            cat = e.get("category", "相关服务")
            grouped.setdefault(cat, []).append(e)
        blocks = []
        for cat, group in grouped.items():
            lines = [f"【{cat}】"]
            for e in group:
                lines.append(f"• {e.get('title', '')}：{e.get('content', '')}")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    # ---- product 字段辅助（从 JSON 内容解析，兼容全角冒号"："） ----
    def _product_brand(self, e: Dict[str, Any]) -> str:
        for line in str(e.get("content", "")).splitlines():
            if line.startswith("品牌："):
                return line.split("：", 1)[1].strip()
        return ""

    def _product_models(self, e: Dict[str, Any]) -> str:
        for line in str(e.get("content", "")).splitlines():
            if line.startswith("型号："):
                return line.split("：", 1)[1].strip()
        return ""

    def _product_price(self, e: Dict[str, Any]) -> str:
        for line in str(e.get("content", "")).splitlines():
            if line.startswith("价格区间："):
                return line.split("：", 1)[1].strip()
        return ""

    def _product_features(self, e: Dict[str, Any]) -> str:
        for line in str(e.get("content", "")).splitlines():
            if line.startswith("主要特点："):
                return line.split("：", 1)[1].strip()
        return ""

    def _product_users(self, e: Dict[str, Any]) -> str:
        for line in str(e.get("content", "")).splitlines():
            if line.startswith("适用人群："):
                return line.split("：", 1)[1].strip()
        return ""

    def _product_rating(self, e: Dict[str, Any]) -> str:
        for line in str(e.get("content", "")).splitlines():
            if line.startswith("推荐指数："):
                return line.split("：", 1)[1].strip()
        return ""

    def _hybrid_ranked(self, domain: str, expanded: str, raw_query: str) -> Union[List[Dict[str, Any]], object, None]:
        """混合精排路：BM25 宽池 top-20 + bge-m3 稠密 top-20 → RRF 融合 → reranker 精排。

        三态返回（生产纪律）：
          - entry 列表：应答（reranker/dense 置信信号达标）
          - _REJECT 哨兵：门禁判定拒答（双低置信 → ood），调用方返回空串，**不再回落 BM25**
            （数据：BM25 高分也可能是词面巧合，如"地球外飞行交通工具"命中手机规格 3.6 分）
          - None：混合路不可用（ENABLE_HYBRID=False / 模型缺失异常），调用方回落纯 BM25
        BM25 宽池空时：不短路，改走纯稠密路（dense top-20），置信判定同前。
        """
        if not ENABLE_HYBRID:
            return None
        fuzzy = self._bm25_hits(domain, expanded, top_k=HYBRID_BM25_TOP_K)
        hybrid = _get_hybrid_retriever(self._kb_path)
        if hybrid is None:
            return None
        title2entry = {e.get("title", ""): e for e in self._data.get(domain, [])}
        result = hybrid.rank(domain, raw_query, [(1.0, e) for e in fuzzy], title2entry)
        if result is None:
            return None
        ranked_titles, rr_top1, dense_top1 = result
        entries = [title2entry[t] for t in ranked_titles if t in title2entry]
        if not entries:
            return None
        # 应答门禁：任一强信号达标即答；双低置信 → 拒答（升级人工）
        if rr_top1 >= RERANK_CONF_FLOOR or dense_top1 >= DENSE_OOD_FLOOR:
            return entries
        return _REJECT

    def retrieve(self, domain: str, query: str) -> str:
        """对外检索入口：返回格式化文本，无命中返回空串（→ 触发 _no_answer 升级）。

        2026-08-13：查询扩展层——先对 query 做同义/上下位扩展（只增不替换），
        再走检索，让"叫法不同但同一物"的新实体词也能命中。不碰 agent 决策。
        2026-08-16：检索链升级——精确预检 → 混合精排（BM25 宽池+稠密+reranker，
        ENABLE_HYBRID 开关；模型不可用自动回落纯 BM25；双低置信拒答不回落）。
        2026-08-16 A4：双低拒答前若应用层注入了改写器，改写 query 重检一次
        （真阳性语义补全越门禁，ood 改写后仍拒——见 _rewrite_and_retry 注释）。
        """
        with self._lock:
            if domain not in self._data:
                return ""
        expanded = _expand_query(query)
        exact = self._exact_hits(domain, expanded)
        if exact:
            return self._format_result(domain, exact)
        hybrid_entries = self._hybrid_ranked(domain, expanded, query)
        if hybrid_entries is _REJECT:
            retried = _rewrite_and_retry(domain, query, self)
            if retried is _REJECT:
                return ""
            return self._format_result(domain, retried)
        if isinstance(hybrid_entries, list):
            return self._format_result(domain, hybrid_entries)
        fuzzy = self._bm25_hits(domain, expanded)
        if fuzzy:
            return self._format_result(domain, fuzzy)
        return ""

    def retrieve_top_title(self, domain: str, query: str) -> Optional[str]:
        """环外只读：返回本次检索命中的 top-1 条目标题（供评估层算精确率@1），无命中返回 None。

        与 retrieve() 共用同一检索判定（精确优先→混合精排→BM25），
        保证"评估看到的就是 agent 检索到的"。只读，不碰 agent 决策逻辑。
        """
        titles = self.retrieve_titles(domain, query, top_k=1)
        return titles[0] if titles else None

    def retrieve_titles(self, domain: str, query: str, top_k: int = 3) -> List[str]:
        """环外只读：返回本次检索命中的条目标题列表（保留命中顺序），无命中返回 []。

        与 retrieve() / retrieve_top_title() 共用同一检索判定（精确优先→混合精排→BM25，
        含 A4 改写重检），保证"引用标注的就是 agent 检索到的"（A6 引用溯源）。
        top_k 截断命中数量；只读，不碰 agent 决策逻辑，不触发 LLM。
        """
        with self._lock:
            if domain not in self._data:
                return []
        expanded = _expand_query(query)
        exact = self._exact_hits(domain, expanded)
        if exact:
            return [str(e.get("title") or "") for e in exact[:top_k] if e.get("title")]
        hybrid_entries = self._hybrid_ranked(domain, expanded, query)
        if hybrid_entries is _REJECT:
            retried = _rewrite_and_retry(domain, query, self)
            if retried is _REJECT:
                return []
            hybrid_entries = retried
        if isinstance(hybrid_entries, list):
            return [str(e.get("title") or "") for e in hybrid_entries[:top_k] if e.get("title")]
        fuzzy = self._bm25_hits(domain, expanded)
        if fuzzy:
            return [str(e.get("title") or "") for e in fuzzy[:top_k] if e.get("title")]
        return []


# 模块级单例（懒加载，全局共享一份索引）
_retriever: Optional[KBRetriever] = None
_retriever_lock = threading.Lock()

# _hybrid_ranked 三态返回的拒答哨兵（区别于 None=模型不可用回落）
_REJECT = object()

# 混合检索器单例（懒加载：构造只读 KB 零成本，模型在首次 rank() 时才加载）
_hybrid_retriever: Any = None
_hybrid_lock = threading.Lock()


def _get_hybrid_retriever(kb_path: str):
    """懒加载 HybridRetriever；任何异常 → None（调用方回落纯 BM25，线上不挂）。"""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        with _hybrid_lock:
            if _hybrid_retriever is None:
                try:
                    from hybrid_retriever import get_hybrid
                    _hybrid_retriever = get_hybrid(kb_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[kb] 混合检索初始化失败，禁用: {e}")
                    _hybrid_retriever = None
    return _hybrid_retriever


# ===== A4 自适应检索：拒答前 query 改写重检（2026-08-16） =====
# 背景：KB 601 扩容后 95 金标中 8 条真阳性 dense 落在 [0.50,0.55) 临界区被误拒
# （mt-206 满赠 / mt-207 订单查询等），而 ood 假阳性（快艇 0.5278 / 太空飞船 0.5177）
# 与该区间重叠——单维降门禁无分隔度（数据否决）。改为：双低拒答前用 LLM 改写
# query（补全口语/业务术语）再检一次——真阳性改写后语义补全分数上推越过门禁，
# ood 改写后仍与 KB 无关分数不变，改写增益天然区分真假阳性。
# 层约束：检索层不 import LLM——改写器由应用层（make_graph）注入，评估脚本
# 不注入即保持纯检索口径（评估一致性）。
_query_rewriter = None  # Callable[[domain, query], str|None]
_query_rewriter_lock = threading.Lock()


def set_query_rewriter(fn) -> None:
    """注入 query 改写器（应用层启动时调用）。fn(domain, query) -> 改写query 或 None。"""
    global _query_rewriter
    with _query_rewriter_lock:
        _query_rewriter = fn


def clear_query_rewriter() -> None:
    """清除改写器（评估/测试隔离用）。"""
    global _query_rewriter
    with _query_rewriter_lock:
        _query_rewriter = None


def _rewrite_and_retry(domain: str, query: str, retriever: "KBRetriever"):
    """拒答前的一次性改写重检：返回 entry 列表（放行）或 _REJECT（维持拒答）。

    只重检一次；改写失败/无变化/仍双低 → 维持原拒答（升级人工兜底不变）。
    """
    fn = _query_rewriter
    if fn is None:
        return _REJECT
    try:
        rewritten = fn(domain, query)
    except Exception as e:  # noqa: BLE001  改写器异常不阻断拒答主路径
        logger.warning("query 改写器异常，维持拒答: %s", e)
        return _REJECT
    if not rewritten or not str(rewritten).strip() or str(rewritten).strip() == query.strip():
        return _REJECT
    rewritten = str(rewritten).strip()
    expanded = _expand_query(rewritten)
    exact = retriever._exact_hits(domain, expanded)
    if exact:
        return exact
    retried = retriever._hybrid_ranked(domain, expanded, rewritten)
    if retried is _REJECT or retried is None:
        return _REJECT
    return retried



def get_retriever() -> KBRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = KBRetriever()
    return _retriever


def retrieve(domain: str, query: str) -> str:
    """便捷函数：等价 get_retriever().retrieve(domain, query)。"""
    return get_retriever().retrieve(domain, query)


def retrieve_titles(domain: str, query: str, top_k: int = 3) -> List[str]:
    """便捷函数：等价 get_retriever().retrieve_titles(domain, query, top_k)。"""
    return get_retriever().retrieve_titles(domain, query, top_k=top_k)