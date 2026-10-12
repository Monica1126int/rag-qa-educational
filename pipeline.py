"""RAG 完整流程封装，支持三种检索配置。"""
from typing import List, Tuple

import config
from src.chunking import load_documents, split_chunks
from src.generator import Generator
from src.models import Embedder, build_reranker
from src.retriever import HybridRetriever, VectorRetriever


class RAGPipeline:
    """三种检索配置：

    - vector        ：纯向量检索
    - hybrid        ：混合检索（BM25 + 向量，RRF）
    - hybrid_rerank ：混合检索 + 交叉编码器重排序
    """
    MODES = ("vector", "hybrid", "hybrid_rerank")

    def __init__(self, mode: str, generator: Generator):
        assert mode in self.MODES, f"mode 必须是 {self.MODES}"
        self.mode = mode
        self.generator = generator

        docs = load_documents(config.DATA_FILE)
        self.chunks = split_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        self.embedder = Embedder(config.EMBEDDING_MODEL)
        self.vector = VectorRetriever(self.chunks, self.embedder)
        self.hybrid = HybridRetriever(self.chunks, self.embedder)
        self.reranker = build_reranker(self.embedder, config.RERANK_MODEL) \
            if mode == "hybrid_rerank" else None

    def retrieve(self, query: str, top_k: int = config.TOP_K) -> List[Tuple[int, float]]:
        if self.mode == "vector":
            return self.vector.retrieve(query, top_k)

        # hybrid 与 hybrid_rerank 都先做混合检索，多召回一些候选项
        candidates = self.hybrid.retrieve(query, top_k=top_k * 2)
        if self.mode == "hybrid":
            return candidates[:top_k]

        # 用交叉编码器对候选项重排序
        cand_texts = [self.chunks[i] for i, _ in candidates]
        scores = self.reranker.score(query, cand_texts)
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])[:top_k]
        return [(candidates[i][0], float(scores[i])) for i in order]

    def answer(self, query: str, top_k: int = config.TOP_K):
        hits = self.retrieve(query, top_k)
        context = "\n".join(self.chunks[i] for i, _ in hits)
        answer = self.generator.generate(query, context)
        return {"hits": hits, "context": context, "answer": answer}
