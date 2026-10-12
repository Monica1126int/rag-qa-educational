# -*- coding: utf-8 -*-
"""环境自检：不花钱/花钱各一步，定位问题出在哪一层。

用法: python check_env.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("=" * 60)
print("1. 依赖")
print("=" * 60)
for m in ["openai", "sentence_transformers", "rank_bm25", "jieba", "numpy", "dotenv"]:
    try:
        __import__(m)
        print(f"   OK    {m}")
    except Exception as e:
        print(f"   FAIL  {m}: {e}")

print()
print("=" * 60)
print("2. 配置（.env）")
print("=" * 60)
import config

key = config.DEEPSEEK_API_KEY
print(f"   API Key: {'已设置，长度 %d，前缀 %s' % (len(key), key[:6]) if key else '❌ 未设置'}")
print(f"   Base URL: {config.DEEPSEEK_BASE_URL}")
print(f"   Model   : {config.DEEPSEEK_MODEL}")

print()
print("=" * 60)
print("3. DeepSeek API 连通性（发一次极短请求，消耗约几十 token）")
print("=" * 60)
try:
    from src.generator import Generator

    g = Generator(config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL)
    resp = g.client.chat.completions.create(
        model=g.model,
        messages=[{"role": "user", "content": "只回复两个字：可用"}],
        temperature=0,
    )
    print("   ✅ API 可用，模型回复：", resp.choices[0].message.content.strip())
except Exception as e:
    print(f"   ❌ API 调用失败：{type(e).__name__}: {e}")

print()
print("=" * 60)
print("4. 本地向量模型 bge-small-zh-v1.5")
print("=" * 60)
try:
    from src.models import Embedder

    emb = Embedder(config.EMBEDDING_MODEL)
    v = emb.encode(["测试句子"])
    print(f"   ✅ 加载成功，向量维度 {v.shape}")
except Exception as e:
    print(f"   ❌ 加载失败：{type(e).__name__}: {e}")

print()
print("=" * 60)
print("5. 本地重排序模型 bge-reranker-base")
print("=" * 60)
try:
    from src.models import build_reranker, Embedder

    _emb = Embedder(config.EMBEDDING_MODEL)
    r = build_reranker(_emb, config.RERANK_MODEL)
    s = r.score("毕业需要多少学分", ["本科生须修满160学分方可毕业。", "期末考试不及格可补考。"])
    print(f"   打分模式：{r.source}，示例打分 {[round(float(x), 3) for x in s]}")
except Exception as e:
    print(f"   ❌ 加载失败：{type(e).__name__}: {e}")
    print("      （通常是没下载到本地且无法联网）")
