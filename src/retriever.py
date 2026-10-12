"""检索器：纯向量检索 与 混合检索（BM25 + 向量，RRF 融合）。"""
from typing import List, Tuple

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from .models import Embedder


class VectorRetriever:
    """A 组：纯向量检索（稠密语义检索）。"""

    def __init__(self, chunks: List[str], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings = embedder.encode(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q = self.embedder.encode([query])[0]
        sims = np.dot(self.embeddings, q)  # 向量已归一化，点积即余弦相似度
        order = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in order]


class HybridRetriever:
    """B 组：混合检索 = BM25（稀疏，字面匹配）+ 向量（稠密，语义匹配），RRF 融合。"""

    def __init__(self, chunks: List[str], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.embeddings = embedder.encode(chunks)
        self.tokenized = [list(jieba.cut(c)) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    @staticmethod
    def _rrf(rankings: List[np.ndarray], k: int = 60):
        """倒数排名融合（Reciprocal Rank Fusion）。"""
        scores = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        # 稠密排名
        q = self.embedder.encode([query])[0]
        sims = np.dot(self.embeddings, q)
        dense_rank = np.argsort(-sims)
        # 稀疏排名
        qtok = list(jieba.cut(query))
        sparse_scores = self.bm25.get_scores(qtok)
        sparse_rank = np.argsort(-sparse_scores)
        # RRF 融合
        return self._rrf([dense_rank, sparse_rank])[:top_k]
