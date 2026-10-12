# -*- coding: utf-8 -*-
"""融合策略对比：RRF vs 语义优先的几种替代方案。

动机：v2 实验里纯向量（0.90/1.00）反而优于 RRF 混合检索（0.77/0.87）。
RRF 只使用排名、忽略分数差距；当其中一路（向量）已经接近满分时，
把稀疏路线的排名等权融合进来，会把向量排在第 1 的正确片段挤下去。
本脚本对比几种「以向量排序为主、BM25 只用于补充候选」的方案。

评测口径：
- Recall 只在 30 道可答题上统计（2 道「无依据拒答」题不计入分母）；
- 拒答题单独统计：检索未命中关键串即视为「应拒答」，用拒答率单独报告。

用法: python fusion_variants.py
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

TOP_K = config.TOP_K


def main():
    docs = load_documents(BASE / "data" / "knowledge_v2.txt")
    chunks = split_chunks(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    cases = json.loads((BASE / "data" / "eval_set_v2.json").read_text(encoding="utf-8"))["cases"]
    answerable = [c for c in cases if c["type"] != "无依据拒答"]
    unanswerable = [c for c in cases if c["type"] == "无依据拒答"]

    emb = Embedder(config.EMBEDDING_MODEL)
    E = emb.encode(chunks)
    bm25 = BM25Okapi([list(jieba.cut(c)) for c in chunks])

    def scores(q):
        v = emb.encode([q])[0]
        dense = E @ v                      # 余弦相似度（已归一化）
        sparse = np.asarray(bm25.get_scores(list(jieba.cut(q))), dtype=np.float64)
        return dense, sparse

    def minmax(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)

    # ---- 各种融合策略：输入 dense/sparse 分数，输出按顺序排列的索引 ----
    def s_vector(d, s):
        return np.argsort(-d)

    def s_rrf(d, s, k=60):
        dr, sr = np.argsort(-d), np.argsort(-s)
        sc = {}
        for ranking in (dr, sr):
            for rank, idx in enumerate(ranking):
                sc[int(idx)] = sc.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
        return np.array([i for i, _ in sorted(sc.items(), key=lambda x: -x[1])])

    def s_union_vecfirst(d, s, n=5):
        """向量 Top-n ∪ BM25 Top-n，统一按向量相似度排序（BM25 只负责补充候选）。"""
        cand = list(dict.fromkeys(list(np.argsort(-d)[:n]) + list(np.argsort(-s)[:n])))
        return np.array(sorted(cand, key=lambda i: -d[i]))

    def s_union_vecfirst_n10(d, s):
        return s_union_vecfirst(d, s, n=10)

    def s_weighted(d, s, w=0.8):
        """加权融合：w×归一化向量分 + (1−w)×归一化 BM25 分。"""
        return np.argsort(-(w * minmax(d) + (1 - w) * minmax(s)))

    def s_bm25_backfill(d, s, n=5):
        """以向量 Top-n 为骨架，仅当骨架不足 n 条时才用 BM25 补位（本实验等价于向量）。"""
        return np.array(list(np.argsort(-d)[:n]))

    strategies = [
        ("vector（纯向量）", s_vector),
        ("RRF（k=60，原实现）", s_rrf),
        ("RRF + 候选池各前5", lambda d, s: s_rrf(d, s)[:0] if False else _rrf_pool(d, s, 5)),
        ("加权融合 w=0.8（向量为主）", s_weighted),
        ("加权融合 w=0.5（等权）", lambda d, s: s_weighted(d, s, 0.5)),
        ("并集按向量排序（各前5）", s_union_vecfirst),
        ("并集按向量排序（各前10）", s_union_vecfirst_n10),
    ]

    print(f"知识库 {len(chunks)} 个 chunk；可答题 {len(answerable)} 道；拒答题 {len(unanswerable)} 道")
    print(f"评测：Recall@1/@3 仅在可答题上统计，Top-K = {TOP_K}\n")
    print("=" * 84)
    print(f"{'融合策略':<30s} | {'Recall@1':>8s} | {'Recall@3':>8s} | {'拒答题未命中率':>12s}")
    print("=" * 84)

    rows = []
    for name, fn in strategies:
        h1 = h3 = 0
        for c in answerable:
            d, s = scores(c["q"])
            order = fn(d, s)
            texts = [chunks[i] for i in order[:TOP_K]]
            ranks = [r for r, t in enumerate(texts, 1) if c["key"] in t]
            h1 += 1 if 1 in ranks else 0
            h3 += 1 if any(r <= 3 for r in ranks) else 0
        # 拒答题：检索未命中关键串 → 视为正确拒答的前提
        abst = 0
        for c in unanswerable:
            d, s = scores(c["q"])
            order = fn(d, s)
            texts = [chunks[i] for i in order[:TOP_K]]
            abst += 1 if not any(c["key"] in t for t in texts) else 0
        r1, r3 = h1 / len(answerable), h3 / len(answerable)
        ar = abst / len(unanswerable) if unanswerable else float("nan")
        rows.append((name, r1, r3, ar))
        print(f"{name:<30s} | {r1:>8.2f} | {r3:>8.2f} | {ar:>12.2f}")

    best = max(rows, key=lambda r: (r[1], r[2]))
    print("=" * 84)
    print(f"最佳策略：{best[0]}  (Recall@1={best[1]:.2f}, Recall@3={best[2]:.2f})")

    out = BASE / "results_fusion.json"
    out.write_text(json.dumps(
        [{"strategy": n, "recall@1": round(a, 4), "recall@3": round(b, 4), "abstain_rate": round(c, 4)}
         for n, a, b, c in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    print("已保存:", out.name)


def _rrf_pool(d, s, pool, k=60):
    dr, sr = np.argsort(-d)[:pool], np.argsort(-s)[:pool]
    sc = {}
    for ranking in (dr, sr):
        for rank, idx in enumerate(ranking):
            sc[int(idx)] = sc.get(int(idx), 0.0) + 1.0 / (k + rank + 1)
    return np.array([i for i, _ in sorted(sc.items(), key=lambda x: -x[1])])


if __name__ == "__main__":
    main()
