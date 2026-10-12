# -*- coding: utf-8 -*-
"""参数扫描：考察 RRF 融合的候选池大小、TOP_K 对三组检索配置的影响。

背景：v2 实验里 hybrid 反而低于纯向量。怀疑原因是 RRF 把两路「全库排名」
都纳入融合，稀疏排名靠前的无关条款会通过 1/(k+rank) 挤掉真正相关的片段。

用法: python tune_rrf.py            # 需要已下载向量模型，可离线跑
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config  # noqa: E402
from src.chunking import load_documents, split_chunks  # noqa: E402
from src.models import Embedder  # noqa: E402

import jieba  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402


def load_eval(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def main():
    kb = BASE / "data" / "knowledge_v2.txt"
    cases = load_eval(BASE / "data" / "eval_set_v2.json")

    docs = load_documents(kb)
    chunks = split_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"知识库: {kb.name}  ({len(docs)} 条规章 → {len(chunks)} 个 chunk)")
    print(f"测试集: {len(cases)} 题\n")

    emb = Embedder(config.EMBEDDING_MODEL)
    E = emb.encode(chunks)
    tokenized = [list(jieba.cut(c)) for c in chunks]
    bm25 = BM25Okapi(tokenized)

    def dense_rank(q):
        v = emb.encode([q])[0]
        return np.argsort(-(E @ v))

    def sparse_rank(q):
        return np.argsort(-bm25.get_scores(list(jieba.cut(q))))

    def rrf(rankings, k=60, pool=None):
        scores = {}
        for ranking in rankings:
            r = ranking if pool is None else ranking[:pool]
            for rank, idx in enumerate(r):
                scores[int(idx)] = scores.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: -x[1])

    def recall(order_fn, top_k=5):
        h1 = h3 = 0
        for c in cases:
            if c["type"] == "无依据拒答":
                continue
            idxs = [i for i, _ in order_fn(c["q"])[:top_k]]
            texts = [chunks[i] for i in idxs]
            ranks = [r for r, t in enumerate(texts, 1) if c["key"] in t]
            h1 += 1 if 1 in ranks else 0
            h3 += 1 if any(r <= 3 for r in ranks) else 0
        n = sum(1 for c in cases if c["type"] != "无依据拒答")
        return h1 / n, h3 / n

    print("=" * 76)
    print("A. 纯向量 vs 混合检索（不同 RRF 候选池大小 pool）")
    print("=" * 76)
    print(f"{'配置':<34s} | {'Recall@1':>8s} | {'Recall@3':>8s}")
    r1, r3 = recall(lambda q: [(i, s) for i, s in zip(dense_rank(q), np.arange(len(chunks)))])
    print(f"{'vector（纯向量）':<34s} | {r1:>8.2f} | {r3:>8.2f}")

    for pool in [None, 50, 30, 20, 15, 10, 5]:
        label = "hybrid（RRF，全库排名）" if pool is None else f"hybrid（RRF，两路各取前 {pool}）"
        r1, r3 = recall(lambda q, p=pool: rrf([dense_rank(q), sparse_rank(q)], pool=p))
        print(f"{label:<34s} | {r1:>8.2f} | {r3:>8.2f}")

    print()
    print("=" * 76)
    print("B. RRF 的 k 参数（候选池 = 两路各前 20）")
    print("=" * 76)
    print(f"{'RRF k':<34s} | {'Recall@1':>8s} | {'Recall@3':>8s}")
    for kk in [1, 5, 10, 20, 60, 100]:
        r1, r3 = recall(lambda q, k=kk: rrf([dense_rank(q), sparse_rank(q)], k=k, pool=20))
        print(f"{'k = ' + str(kk):<34s} | {r1:>8.2f} | {r3:>8.2f}")

    print()
    print("=" * 76)
    print("C. 逐题对比：vector vs hybrid(全库) —— 定位 hybrid 的失败题")
    print("=" * 76)
    v_order = lambda q: [(i, 0) for i in dense_rank(q)]
    h_order = lambda q: rrf([dense_rank(q), sparse_rank(q)])
    print(f"{'id':>3} {'类型':<8} {'vector':>8} {'hybrid':>8}  问题")
    worse = []
    for c in cases:
        if c["type"] == "无依据拒答":
            continue
        def rank_of(order_fn):
            idxs = [i for i, _ in order_fn(c["q"])[:5]]
            texts = [chunks[i] for i in idxs]
            rs = [r for r, t in enumerate(texts, 1) if c["key"] in t]
            return rs[0] if rs else 0
        rv, rh = rank_of(v_order), rank_of(h_order)
        mark = ""
        if rv == 1 and rh != 1:
            mark = "  ← hybrid 变差"
            worse.append(c["id"])
        print(f"{c['id']:>3} {c['type']:<8} {('rank%d' % rv) if rv else 'miss':>8} "
              f"{('rank%d' % rh) if rh else 'miss':>8}  {c['q'][:40]}{mark}")
    print(f"\nhybrid 把 Top-1 弄丢的题目 id: {worse}")


if __name__ == "__main__":
    main()
