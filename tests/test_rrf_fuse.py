#!/usr/bin/env python3
"""rrf_fuse 加权融合密封测试（RAG 第一批配套，2026-09-16）。

具名风险（AGENTS.md §5"每个测试对应具名风险"）：
  F1 参数无意识漂移——融合参数经 217 组件评估 + W4 门禁定版（k=20/w=1.5），
     常量被悄悄改动将绕过门禁直接改变线上排序。锁定常量，变更必须过门禁。
  F2 生产/评估口径漂移——历史上真实发生过（SCORE_THRESHOLD 两路口径差导致
     sweep 与 details 不一致）。两处 rrf_fuse 必须对同一输入产出同一输出。
  F3 加权语义错误——w_dense 提权稠密路的公式写错（如权重乘错路/秩偏移 off-by-one）
     不会被现有任何测试发现（融合前无覆盖）。
  F4 回滚开关失效——声称"改回 w=1.0 即恢复等权 RRF"，必须实际成立。
  F5 GPU 段并发穿透——并发 forward 激活显存叠加挤爆 8GB 触发 WDDM 分页悬崖
     （bench_20260916：32 并发 0.13 QPS / 单 batch 121s / 显存 7.8GB），
     _gpu_lock 若被误删/缩窄到只包 predict，密封测试必须当场红。

密封纪律：纯函数级，无模型/网络/真实时钟；eval 脚本导入含 os.chdir 副作用，
由 fixture 保存恢复（AGENTS.md §3.6 全局副作用配对清理）。F5 用 mock 模型
（无真实 GPU），靠共享 busy 标志检测互斥违反。
"""
import os
import sys
import threading
import time

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

from hybrid_retriever import RRF_K, RRF_W_DENSE, rrf_fuse  # noqa: E402


@pytest.fixture(scope="module")
def eval_rrf():
    """导入评估侧实现（导入时 os.chdir，配对恢复；模块缓存后不再重复副作用）。"""
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(PACKAGE_ROOT, "eval", "kb_v2"))
    import eval_retrieval_rerank as mod
    os.chdir(cwd)
    return mod


def test_f1_params_locked():
    """F1：融合参数锁定（k=20, w_dense=1.5）——变更必须过 217 组件评估 + W4 门禁。"""
    assert RRF_K == 20, "RRF_K 变更 = 模型行为变更，须走 W4 门禁（见 AGENTS.md §3.1）"
    assert RRF_W_DENSE == 1.5, "RRF_W_DENSE 变更 = 模型行为变更，须走 W4 门禁"


def test_f2_eval_pipeline_same_fusion(eval_rrf):
    """F2：生产/评估同口径——两处 rrf_fuse 对同一输入输出完全一致。"""
    bm25_hits = [(9.5, "A"), (8.1, "B"), (7.2, "C"), (0.4, "D")]
    dense_hits = [(0.9, "E"), (0.8, "B"), (0.7, "F"), (0.6, "A")]
    for top_k in (5, 20):
        assert rrf_fuse(bm25_hits, dense_hits, top_k=top_k) == \
            eval_rrf.rrf_fuse(bm25_hits, dense_hits, top_k=top_k), \
            f"生产/评估融合口径漂移（top_k={top_k}）"


def test_f3_weighted_dense_semantics():
    """F3：加权语义——bm25/dense 同候选同秩时，dense 路权重使其胜出。"""
    bm25_only = [(5.0, "BM")]            # 仅 BM25 路，rank0
    dense_only = [(0.9, "DN")]           # 仅 dense 路，rank0
    fused = rrf_fuse(bm25_only + [(1.0, "X")], dense_only + [(0.5, "X")])
    # DN（dense rank0）应排在 BM（bm25 rank1）之前：1.5/(k+1) > 1/(k+2)
    assert fused.index("DN") < fused.index("BM")
    # X 两路均 rank1，得分 = (1 + 1.5)/(k+2)，应高于任一单路 rank0？
    # 1/(k+1)=0.0476 vs 2.5/22=0.1136 → X 应居首
    assert fused[0] == "X"


def test_f4_rollback_equal_weights():
    """F4：回滚开关——w=1.0 时退化为纯排名 RRF（手工算例对拍）。"""
    bm25_hits = [(5.0, "A"), (4.0, "B")]
    dense_hits = [(0.9, "C"), (0.8, "A")]
    # 等权手工计算（k=20）：A = 1/21 + 1/22 ≈ 0.0931；B = 1/22 ≈ 0.0455；C = 1/21 ≈ 0.0476
    scores = {}
    for rank, (_, t) in enumerate(bm25_hits):
        scores[t] = scores.get(t, 0.0) + 1.0 / (20 + rank + 1)
    for rank, (_, t) in enumerate(dense_hits):
        scores[t] = scores.get(t, 0.0) + 1.0 / (20 + rank + 1)  # w=1.0
    expected = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    # 用 hybrid_retriever 的实现以 w=1.0 口径复算：直接 monkeypatch 权重不可行
    # （闭包引用模块常量），改为进程内临时改常量再恢复（配对清理）。
    import hybrid_retriever as hr
    old = hr.RRF_W_DENSE
    try:
        hr.RRF_W_DENSE = 1.0
        assert hr.rrf_fuse(bm25_hits, dense_hits) == expected
    finally:
        hr.RRF_W_DENSE = old


class _GpuBusyReranker:
    """mock reranker：进入 predict 即置共享 busy 标志，若已 busy 则记录违反。"""

    def __init__(self, state):
        self._state = state

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        with self._state["m"]:
            if self._state["busy"]:
                self._state["violations"] += 1
            self._state["busy"] = True
        time.sleep(0.02)  # 模拟 forward，给并发线程留重叠窗口
        with self._state["m"]:
            self._state["busy"] = False
        return [0.5] * len(pairs)


class _GpuBusyDense:
    """mock dense：与 reranker 共享同一 busy 标志（锁必须覆盖整个 GPU 段）。"""

    def __init__(self, state, titles):
        self._state = state
        self._titles = titles
        self._vectors = {"__domain__": True}  # rank() 的域存在性检查

    def search(self, domain, query, top_k=20):
        with self._state["m"]:
            if self._state["busy"]:
                self._state["violations"] += 1
            self._state["busy"] = True
        time.sleep(0.01)
        with self._state["m"]:
            self._state["busy"] = False
        return [(0.9 - i * 0.01, t) for i, t in enumerate(self._titles)]


def test_f5_gpu_section_mutual_exclusion():
    """F5：GPU 段互斥——并发 rank() 的 dense/rerank 调用不得重叠（busy 标志零违反）。"""
    import hybrid_retriever as hr

    kb_path = os.path.join(PACKAGE_ROOT, "data", "knowledge_base_v2.json")
    h = hr.HybridRetriever(kb_path=kb_path)
    titles = [f"条目{i}" for i in range(5)]
    title2entry = {t: {"title": t, "content": f"{t}的内容"} for t in titles}
    state = {"m": threading.Lock(), "busy": False, "violations": 0}
    # 直接注入 mock，绕过模型懒加载（密封纪律：不加载真实 GPU 模型）
    h._dense = _GpuBusyDense(state, titles)
    h._reranker = _GpuBusyReranker(state)

    bm25_ranked = [(5.0, title2entry[t]) for t in titles[:3]]
    n_threads = 4
    barrier = threading.Barrier(n_threads)

    def worker(i):
        barrier.wait()  # 同时起跑，最大化重叠概率
        r = h.rank("__domain__", f"查询{i}", bm25_ranked, title2entry)
        assert r is not None, "mock 路径 rank() 不应回落 None"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["violations"] == 0, (
        f"GPU 段并发穿透 {state['violations']} 次——_gpu_lock 缺失或覆盖不全，"
        "WDDM 显存叠加悬崖风险回归（bench_20260916）"
    )
