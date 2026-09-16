#!/usr/bin/env python3
"""
混合检索模块（生产版，kb-v2 上线配套，2026-08-16）。

背景：kb-v2 规模扩大后（277→601 条）BM25 相对退化（模板词污染、口语 query 词不匹配）。
P1/P2 评估（95 条金标，2026-08-16 时点）数据支撑：
  BM25-only      R@1 0.374 → RRF 混合 0.447 → +bge-reranker-v2-m3 0.468
  hard 档 R@1 +39%、ambiguous 档 +75%（reranker 语义价值区），easy 档 -14%（有回归，
  已由"精确预检优先"在调用方挡住大部分）。
2026-09-16 调参 + 金标修正后（217 条金标，详见 ai-docs/w5-rag-batch1.md）：
  BM25 0.357 / 加权 RRF 0.493 / +rerank 0.482（R@1）；rerank R@5 0.790（池质量）。
详见 eval/kb_v2/eval_retrieval_rerank.py 与 eval_retrieval_rerank_details.json。

设计约束（生产纪律）：
  1. 懒加载单例：模型首次检索才加载（服务启动零成本），全局共享一份。
  2. 失败回落：模型缺失/加载/推理异常 → rank() 返回 None，调用方回退 BM25-only，线上不挂。
  3. GPU 优先，无 CUDA 回落 CPU（慢但可用）。
  4. 只负责"排序"，不做 ood 门禁：BM25 top-20 全空由调用方短路（保 ood 纪律）。
  5. 稠密路复用 eval/kb_v2/dense_retriever.py（单点维护，改动需同步评估侧）。
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

# 稠密路：复用评估侧 DenseRetriever（同一份实现，避免生产/评估漂移）
_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL_KB_DIR = os.path.join(_HERE, "eval", "kb_v2")
if _EVAL_KB_DIR not in sys.path:
    sys.path.insert(0, _EVAL_KB_DIR)

from dense_retriever import DenseRetriever  # noqa: E402

# bge-reranker-v2-m3 本地快照路径（环境变量注入，不入库；2026-08-16 smoke：加载 4.0s / 20 候选 0.63s）
BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", "")

# 稠密向量缓存（KB 未变时复用，避免每次重启重编码）
DENSE_CACHE_PATH = os.path.join(_HERE, "data", "knowledge_base_v2_bge_m3_vectors.npz")

# 与评估脚本同口径的融合/重排参数
# 2026-09-16 调参（实验 B 消融定版，217 金标 HIT@1 +3.2pp）：
#   k 60→20、稠密路加权 w_dense=1.5（稠密路质量高于 BM25，R@1 0.422 vs 0.343，提权收益实锤）。
#   回滚 = 改回 RRF_K=60 / RRF_W_DENSE=1.0（2026-09-16 之前的等权行为）。
RRF_K = 20
RRF_W_DENSE = 1.5    # 稠密路 RRF 权重（1.0 = 原等权）
DENSE_TOP_K = 20       # 稠密路召回宽度（RRF 融合需要宽池）
RERANK_POOL = 20       # RRF 融合后送入 reranker 的池宽
RERANK_TOP_K = 5       # reranker 精排输出


def rrf_fuse(bm25_hits: list, dense_hits: list, top_k: int = RERANK_POOL) -> List[str]:
    """加权 RRF 融合：输入 [(score, title)] 两路（各自有序），输出前 top_k 个 title。

    score(d) = 1.0/(RRF_K + rank_bm25) + RRF_W_DENSE/(RRF_K + rank_dense)
    与 eval_retrieval_rerank.py 同口径，保证生产排序 == 评估排序。
    """
    scores: Dict[str, float] = {}
    for rank, (_, title) in enumerate(bm25_hits):
        scores[title] = scores.get(title, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (_, title) in enumerate(dense_hits):
        scores[title] = scores.get(title, 0.0) + RRF_W_DENSE / (RRF_K + rank + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ranked[:top_k]]


class HybridRetriever:
    """BM25 + bge-m3 稠密 → RRF 融合 → bge-reranker-v2-m3 精排。

    生产用法（由 KBRetriever 调用）：
        ranked = hybrid.rank(domain, query, bm25_ranked, title2entry)
        # ranked: Optional[List[str]]，None = 模型不可用（调用方回退 BM25-only）
    """

    def __init__(self, kb_path: str) -> None:
        self._kb_path = kb_path
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        # GPU 串行门控（2026-09-16）：见 rank() 内注释
        self._gpu_lock = threading.Lock()
        self._dense: Optional[DenseRetriever] = None
        self._reranker = None
        self._load_kb()

    def _load_kb(self) -> None:
        import json
        with open(self._kb_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for domain, entries in raw.items():
            if domain.startswith("_"):
                continue
            self._data[domain] = entries

    # ---- 懒加载模型（异常向上抛，由 rank() 统一回落） ----

    def warm_up(self) -> None:
        """预加载 + 预推理（启动预热专用，warmup.py 调用）。

        1) 显式触发懒加载——exact 命中的查询会被调用方精确预检短路，
           依赖查询走通全链不可靠（冷启动 22s 复现根因，2026-09-15）。
        2) 加载后必须再跑一次真实 rank()：dense 首次 query 编码与 reranker
           首帧 forward（torch/CUDA 上下文、算子初始化）各占秒级，只加载权重
           不够——Triton model warmup 的语义就是"虚拟推理"而非"权重加载"。
        """
        self._get_dense()
        self._get_reranker()
        domain = next(iter(self._data), None)
        entries = self._data.get(domain) or []
        if not entries:
            return
        title2entry = {str(e.get("title") or ""): e for e in entries}
        first = next(iter(title2entry))
        try:
            self.rank(domain, "预热查询", [(1.0, title2entry[first])], title2entry)
        except Exception:  # noqa: BLE001  预推理失败不阻断——rank() 失败回落 BM25-only 已有兜底
            pass

    def _get_dense(self) -> DenseRetriever:
        if self._dense is None:
            with self._lock:
                if self._dense is None:
                    self._dense = DenseRetriever(
                        kb_path=self._kb_path, cache_path=DENSE_CACHE_PATH
                    )
                    self._dense.build_index()
        return self._dense

    def _get_reranker(self):
        if self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._reranker = CrossEncoder(BGE_RERANKER_PATH, device=device)
        return self._reranker

    # ---- 主入口 ----

    def rank(
        self,
        domain: str,
        query: str,
        bm25_ranked: List[Tuple[float, Dict[str, Any]]],
        title2entry: Dict[str, Dict[str, Any]],
        top_k: int = RERANK_TOP_K,
    ) -> Optional[Tuple[List[str], float, float]]:
        """对候选做混合精排，返回 (top_k titles, rerank_top1_score, dense_top1_score)。

        bm25_ranked：[(bm25_score, entry)]，调用方已按分数降序、阈值过滤、截断 top-20；
                     传空列表 = 纯稠密路（BM25 空召回时由调用方决定是否兜底）。
        title2entry：domain 内 title → entry 映射（reranker 打分需要完整条目文本）。
        返回 None = 稠密/重排模型不可用（调用方回落 BM25-only）。
        只负责排序与返回置信分数；是否应答（门禁）由调用方根据分数判定（职责分离）。
        """
        try:
            dense = self._get_dense()
            reranker = self._get_reranker()
        except Exception as e:  # noqa: BLE001 —— 模型缺失/加载失败 → 回落
            print(f"[hybrid] 模型不可用，回落 BM25-only: {e}")
            return None

        try:
            if domain not in dense._vectors:
                return None
            # GPU 串行门控（2026-09-16）：dense 编码 + rerank forward 同一时刻至多一个线程。
            # 根因（bench/reports/bench_20260916.json）：并发 forward 的激活显存叠加挤爆
            # 8GB 显存触发 WDDM 分页悬崖——32 并发实测吞吐崩塌至 0.13 QPS、单 batch
            # 121s、显存 7.8GB；c>=15 间歇发作（P95 64-68s）。串行本身零代价
            # （bench/gpu_profile.py 微基准：8 线程并发 predict 惩罚 1.0x），锁只是把
            # "碰运气的串行"变成"有保证的串行"，显存有界 → 悬崖从构造上消除。
            with self._gpu_lock:
                dense_ranked = dense.search(domain, query, top_k=DENSE_TOP_K)
                dense_top1 = float(dense_ranked[0][0]) if dense_ranked else 0.0
                pool = rrf_fuse(
                    [(s, e.get("title", "")) for s, e in bm25_ranked], dense_ranked,
                    top_k=RERANK_POOL,
                )
                pairs = []
                pool_titles = []
                for t in pool:
                    entry = title2entry.get(t)
                    if entry is not None:
                        pairs.append((query, DenseRetriever._entry_text(entry)))
                        pool_titles.append(t)
                if not pairs:
                    return None
                scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            ranked = sorted(
                zip(scores, pool_titles), key=lambda x: x[0], reverse=True
            )[:top_k]
            if not ranked:
                return None
            rr_top1 = float(ranked[0][0])
            return [t for _, t in ranked], rr_top1, dense_top1
        except Exception as e:  # noqa: BLE001 —— 推理异常 → 回落
            print(f"[hybrid] 推理异常，回落 BM25-only: {e}")
            return None


# 模块级单例（懒加载；构造只读 KB 不加载模型，成本可忽略）
_hybrid: Optional[HybridRetriever] = None
_hybrid_lock = threading.Lock()


def get_hybrid(kb_path: str) -> Optional[HybridRetriever]:
    """返回全局共享 HybridRetriever；构造失败（KB 不可读）返回 None（调用方纯 BM25）。"""
    global _hybrid
    if _hybrid is None:
        with _hybrid_lock:
            if _hybrid is None:
                try:
                    _hybrid = HybridRetriever(kb_path=kb_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[hybrid] 初始化失败，禁用混合检索: {e}")
                    return None
    return _hybrid
