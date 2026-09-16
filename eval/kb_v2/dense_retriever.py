#!/usr/bin/env python3
"""稠密检索模块：bge-m3 编码 + 余弦相似度检索（P1 混合检索的稠密路）。

- 模型：bge-m3 本地完整快照（路径由环境变量 BGE_M3_PATH 配置，见 env_example.txt）
- GPU 优先（device='cuda'），无 CUDA 回落 CPU
- 向量缓存：KB 编码结果落盘（.npz），KB 未变时复用，避免重复编码
- 检索：query 编码 → 与域内条目向量余弦 → top-k

用法（模块）：
    from dense_retriever import DenseRetriever
    dr = DenseRetriever(kb_path=..., cache_path=...)
    dr.build_index()          # 编码全 KB（幂等，有缓存跳过）
    hits = dr.search(domain, query, top_k=20)  # [(score, title), ...]
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # 模型路径不入库（.env 本地配置），幂等加载

# bge-m3 完整快照路径（环境变量注入；缺失时模型加载失败由调用方失败回落兜底）
BGE_M3_PATH = os.getenv("BGE_M3_PATH", "")

# 稠密检索召回条数（混合融合用，比 BM25 TOP_K=3 宽，交给 RRF 精排）
DENSE_TOP_K = 20


class DenseRetriever:
    """bge-m3 稠密检索器：按 domain 维护条目向量，query 余弦检索。"""

    def __init__(self, kb_path: str, cache_path: Optional[str] = None) -> None:
        self._kb_path = kb_path
        self._cache_path = cache_path
        self._model = None
        self._lock = threading.Lock()
        self._data: Dict[str, List[Dict]] = {}
        self._vectors: Dict[str, np.ndarray] = {}  # domain -> (n, 1024)
        self._titles: Dict[str, List[str]] = {}    # domain -> [title, ...]（与向量行对齐）
        self._load_kb()

    def _load_kb(self) -> None:
        with open(self._kb_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for domain, entries in raw.items():
            if domain.startswith("_"):
                continue
            self._data[domain] = entries

    def _get_model(self):
        """懒加载 bge-m3（GPU 优先）。"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self._model = SentenceTransformer(BGE_M3_PATH, device=device)
        return self._model

    @staticmethod
    def _entry_text(entry: Dict) -> str:
        """条目编码文本 = 标题 + 内容 + 关键词（与 BM25 检索文本对齐，保证两路可比）。"""
        parts = [entry.get("title", ""), entry.get("content", "")]
        kws = entry.get("keywords", [])
        if isinstance(kws, list):
            parts.extend(str(k) for k in kws)
        models = entry.get("models", [])
        if isinstance(models, list):
            parts.extend(str(m) for m in models)
        return " ".join(p for p in parts if p)

    def build_index(self) -> None:
        """编码全 KB（幂等：有缓存且 KB 未变则加载缓存）。"""
        if self._cache_path and os.path.exists(self._cache_path):
            loaded = self._load_cache()
            if loaded:
                return
        model = self._get_model()
        for domain, entries in self._data.items():
            if not entries:
                continue
            texts = [self._entry_text(e) for e in entries]
            vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
            self._vectors[domain] = np.asarray(vecs, dtype=np.float32)
            self._titles[domain] = [e.get("title", "") for e in entries]
        if self._cache_path:
            self._save_cache()

    def _save_cache(self) -> None:
        """缓存：域清单 + 每域向量 + 标题（.npz 单文件）。"""
        os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
        arrays = {}
        for dom in self._vectors:
            arrays[f"vec_{dom}"] = self._vectors[dom]
            arrays[f"titles_{dom}"] = np.array(self._titles[dom], dtype=object)
        arrays["_domains"] = np.array(list(self._vectors.keys()), dtype=object)
        arrays["_kb_hash"] = np.array([self._kb_signature()])
        np.savez(self._cache_path, **arrays)

    def _load_cache(self) -> bool:
        """加载缓存；KB 签名不符则丢弃返回 False。"""
        try:
            data = np.load(self._cache_path, allow_pickle=True)
            if data["_kb_hash"][0] != self._kb_signature():
                return False
            domains = list(data["_domains"])
            for dom in domains:
                self._vectors[dom] = data[f"vec_{dom}"]
                self._titles[dom] = list(data[f"titles_{dom}"])
            return bool(self._vectors)
        except Exception:
            return False

    def _kb_signature(self) -> str:
        """KB 内容签名：条目数 + 标题列表（缓存失效依据）。"""
        sig = []
        for dom, entries in self._data.items():
            sig.append(f"{dom}:{len(entries)}:{','.join(e.get('title', '') for e in entries)}")
        return "|".join(sig)

    def search(self, domain: str, query: str, top_k: int = DENSE_TOP_K) -> List[Tuple[float, str]]:
        """余弦检索：返回 [(score, title), ...] 降序。"""
        if domain not in self._vectors:
            return []
        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True)[0]
        sims = self._vectors[domain] @ q_vec  # (n,) 余弦（已归一化）
        order = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self._titles[domain][i]) for i in order]


def default_cache_path(kb_path: str) -> str:
    """默认缓存路径：与 KB 同目录。"""
    base = os.path.splitext(kb_path)[0]
    return base + "_bge_m3_vectors.npz"


def build_dense_index(kb_path: str, cache_path: Optional[str] = None) -> DenseRetriever:
    """便捷入口：构建（或加载缓存）稠密索引。"""
    dr = DenseRetriever(kb_path=kb_path, cache_path=cache_path or default_cache_path(kb_path))
    dr.build_index()
    return dr
