"""文档加载与分块模块。

将中文教务知识库文本切分为适合检索的 chunk。
"""
from pathlib import Path
from typing import List


def load_documents(path) -> List[str]:
    """读取知识库文本，按空行分隔的段落切分为文档。

    约定：knowledge.txt 中每条规章制度占一段（段间空行分隔）；
    若全文无空行，则退化为按行切分。
    """
    raw = Path(path).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [l.strip() for l in raw.split("\n") if l.strip()]
    return paragraphs


def split_chunks(docs: List[str], chunk_size: int = 200, overlap: int = 40) -> List[str]:
    """对过长文档做滑动窗口分块（按字符），短文档原样保留。"""
    chunks: List[str] = []
    for doc in docs:
        if len(doc) <= chunk_size:
            chunks.append(doc)
            continue
        start = 0
        while start < len(doc):
            chunks.append(doc[start:start + chunk_size])
            if start + chunk_size >= len(doc):
                break
            start += chunk_size - overlap
    return chunks
