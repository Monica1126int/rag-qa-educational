"""向量化（Embedding）与重排序（Reranker）模型封装。

使用 BAAI 的 bge 系列中文模型，首次运行会自动从 HuggingFace（或镜像）下载。

重排序有两套实现：
- CrossEncoderReranker：bge-reranker-base 交叉编码器，效果最好，需要本地模型权重；
- LocalHybridReranker：当交叉编码器不可用（未下载且无法联网）时自动退化的本地重排序，
  用「查询—片段语义相似度 + BM25 字面分」两个信号做 z-score 标准化后等权融合。
  它不是交叉编码器，只作为无法下载模型时的可用替代，实验报告中需注明实际使用的模式。
"""
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class Embedder:
    """稠密向量编码器。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, normalize: bool = True):
        return self.model.encode(texts, normalize_embeddings=normalize)


class CrossEncoderReranker:
    """交叉编码器重排序（推荐）。"""

    source = "cross_encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts):
        pairs = [[query, t] for t in texts]
        return self.model.predict(pairs)


class LocalHybridReranker:
    """本地兜底重排序：语义相似度 + BM25 字面分，z-score 标准化后等权融合。

    仅用于交叉编码器不可用时，保证实验流程可完整跑通。
    """

    source = "local_hybrid"

    def __init__(self, embedder: Embedder, alpha: float = 0.5):
        import jieba

        self.jieba = jieba
        self.embedder = embedder
        self.alpha = alpha  # 语义分权重，剩余权重给 BM25 字面分

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        std = float(x.std())
        if std < 1e-9:  # 分数全相同，退化为零向量
            return np.zeros_like(x)
        return (x - float(x.mean())) / std

    def score(self, query: str, texts):
        # 语义分：查询与候选片段的余弦相似度（向量已归一化，点积即余弦）
        q = self.embedder.encode([query])[0]
        embs = self.embedder.encode(list(texts))
        cos = np.asarray(embs @ q, dtype=np.float64)

        # 字面分：BM25（在候选项内部建索引，只用于排序）
        from rank_bm25 import BM25Okapi

        tokenized = [list(self.jieba.cut(t)) for t in texts]
        try:
            bm25 = np.asarray(BM25Okapi(tokenized).get_scores(list(self.jieba.cut(query))),
                              dtype=np.float64)
        except Exception:
            bm25 = np.zeros(len(texts), dtype=np.float64)

        return self.alpha * self._zscore(cos) + (1 - self.alpha) * self._zscore(bm25)


def build_reranker(embedder: Embedder, model_name: str = "BAAI/bge-reranker-base"):
    """优先加载交叉编码器；失败则退化为本地重排序，并打印所用模式。"""
    try:
        r = CrossEncoderReranker(model_name)
        print(f"   [重排序] 已加载交叉编码器 {model_name}")
        return r
    except Exception as e:
        print(f"   [重排序] 交叉编码器 {model_name} 不可用（{type(e).__name__}），"
              f"退化为本地混合重排序（语义+BM25）")
        return LocalHybridReranker(embedder)
