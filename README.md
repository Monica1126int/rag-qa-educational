# 面向高校教务知识库的 RAG 问答系统

课程设计报告《面向高校教务知识库的 RAG 问答系统设计与初步实现》配套代码。

实现一个中文检索增强生成（RAG）问答原型，并对比三种检索配置的答案质量：

| 配置 | 说明 |
| --- | --- |
| `vector` | 纯向量检索（BGE 稠密语义检索） |
| `hybrid` | 混合检索（BM25 稀疏 + 向量稠密，RRF 融合） |
| `hybrid_rerank` | 混合检索 + 交叉编码器重排序（BGE Reranker） |

## 技术栈

- 生成：DeepSeek 大模型 API（OpenAI 兼容接口，模型 `deepseek-flash`）
- 向量化：`BAAI/bge-small-zh-v1.5`（sentence-transformers）
- 重排序：`BAAI/bge-reranker-base`（CrossEncoder）
- 稀疏检索：`rank-bm25` + `jieba` 中文分词

> 说明：为降低环境依赖、保证可复现并便于理解底层原理，本实现用轻量库直接构建
> 检索组件，其作用等价于 LangChain / LlamaIndex 中的 `Retriever`。如需用
> LangChain 封装，可参考文末「LangChain 对接」一节。

## 环境准备

1. 安装 Python 3.10+，创建虚拟环境并安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

2. 配置 API Key：

   ```bash
   cp .env.example .env      # Windows: Copy-Item .env.example .env
   ```

   用文本编辑器打开生成的 `.env`，**只改第一行**，把你的 Key 粘在等号后面：

   ```
   DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
   ```

   > Key 在 DeepSeek 开放平台 <https://platform.deepseek.com/api_keys> 创建。
   > 注意：DeepSeek 的 API **不赠送免费额度**，新账号需先充值少量余额（本实验
   > 全程约 40 次调用，几角钱即可跑完）。
   >
   > 另外，旧模型名 `deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 停用，
   > 本项目默认使用官方现役模型 `deepseek-flash`（已在 `config.py` 中设好）。
   > 若想改用完全免费的智谱 GLM，只需把 `.env` 里的 `DEEPSEEK_BASE_URL` 改为
   > `https://open.bigmodel.cn/api/paas/v4`、`DEEPSEEK_MODEL` 改为免费模型名，
   > `DEEPSEEK_API_KEY` 填智谱的 Key，代码无需任何改动。

3. 首次运行会自动下载 bge 模型（向量模型约 100MB，重排序模型约 1.1GB）。
   国内直连 HuggingFace 常超时，可在 `.env` 里指定镜像或代理：

   ```
   HF_ENDPOINT=https://hf-mirror.com
   # 或（有本地代理时）
   HTTPS_PROXY=http://127.0.0.1:7897
   HTTP_PROXY=http://127.0.0.1:7897
   ```

   模型下载完成后，可设 `HF_HUB_OFFLINE=1` 完全离线运行。

## 运行

```bash
# 1) 环境自检（先确认 Key 可用、模型能加载，只花一次 API 调用）
python check_env.py

# 2) 完整对比实验：三组配置 × 32 题，含 LLM 裁判
python run_experiment.py --kb data/knowledge_v2.txt --eval data/eval_set_v2.json --judge-all

# 省钱模式：只评测前 5 题
python run_experiment.py --kb data/knowledge_v2.txt --eval data/eval_set_v2.json

# 只测检索、完全不调用大模型（不花钱）
python run_experiment.py --kb data/knowledge_v2.txt --eval data/eval_set_v2.json --no-api
```

实验脚本与会话输出：

| 脚本 | 用途 |
| --- | --- |
| `run_experiment.py` | 主实验：三组配置对比，输出 `results.json` 逐题明细 |
| `run_rerank_crossencoder.py` | 只重跑 C 组（真实交叉编码器），并把结果合并回结果文件 |
| `tune_rrf.py` | RRF 的候选池大小与 k 参数扫描 |
| `fusion_variants.py` | 对比 RRF、加权融合、并集等 6 种融合策略 |
| `check_env.py` | 环境自检（依赖、Key、模型加载） |

## 实验结果（真实运行，2026-10）

知识库 35 条教务规章；测试集 32 题（直接事实 10、改写提问 18、易混淆数字 2、
无依据拒答 2）；Recall 与裁判分只在 30 道可答题上统计，裁判分满分 5 分。

| 配置 | Recall@1 | Recall@3 | LLM 裁判分 | 拒答前提成立率 |
| --- | --- | --- | --- | --- |
| `vector`（纯向量） | **0.90** | **1.00** | **4.75** | 100% |
| `hybrid`（BM25 + 向量，RRF） | 0.77 | 0.87 | 4.44 | 100% |
| `hybrid_rerank`（+ 交叉编码器） | **0.90** | 0.97 | 4.44 | 100% |

**结论与常见直觉相反：**

1. **纯向量检索最好，混合检索反而下降**（Recall@1 从 0.90 掉到 0.77）。
   RRF 只用排名、不看分数差距，当向量检索已能把 90% 的题目排对第 1 位、
   而 BM25 单独使用时 Recall@1 仅 0.67 时，等权融合相当于用明显更弱的一路
   稀释另一路：BM25 抬高了含相同数字的干扰条款，把正确条款挤出 Top-5。
2. **交叉编码器能救回融合损失，但无法超越纯向量**：C 组把 B 组名次下降的
   4 道题全部救回 rank1，Recall@1 恰好追平；但 Recall@3 仍低于 A 组，
   因为重排序只能在候选集合内部调整，**上限由上游召回决定**。
3. **检索指标改善不一定带来答案质量提升**：C 组 Recall@1 比 B 组高 0.13，
   裁判分却同为 4.44。逐题核对发现 C 组有三道题同样命中正确片段，但因 Top-5
   的排列与词条不同，生成的答案反不如 A 组完整——**“命中”只是必要条件**。

实践建议：**稀疏检索的接入方式比是否接入更重要**。稠密检索接近上限时，
应把 BM25 用作候选补充（并集后按稠密分数排序），而不是当作等价的排序信号去融合。

> 详细分析、图表与讨论见课程报告《面向高校教务知识库的 RAG 问答系统设计与初步实现》。

## 自定义知识库

把 `data/knowledge_v2.txt` 替换成你自己的文档即可，按「一段一条」的格式组织，
段间用空行分隔；同时按 `data/eval_set_v2.json` 的格式准备测试集
（`q` 问题、`key` 判定命中的关键信息串、`type` 题型）。分块大小与 Top-K
可在 `config.py` 中调整。

## 代码结构

```
rag_qa/
├── config.py                    # API Key、模型名、网络代理与超参
├── pipeline.py                  # RAG 流程组装（三种检索配置）
├── run_experiment.py            # 主实验 + 评测（输出逐题明细）
├── run_rerank_crossencoder.py   # 单独重跑 C 组并合并结果
├── tune_rrf.py                  # RRF 参数扫描
├── fusion_variants.py           # 6 种融合策略对比
├── check_env.py                 # 环境自检
├── data/
│   ├── knowledge_v2.txt         # 35 条知识库（正式实验用）
│   ├── eval_set_v2.json         # 32 题测试集（正式实验用）
│   ├── knowledge.txt            # 第一轮 10 条知识库（对照）
│   └── eval_set.json            # 第一轮 10 题测试集（对照）
└── src/
    ├── chunking.py              # 文档加载与分块
    ├── models.py                # 向量化 + 交叉编码器重排序
    ├── retriever.py             # 向量检索 / 混合检索（RRF）
    └── generator.py             # DeepSeek 生成
```

## 复现说明

- 生成与裁判使用 DeepSeek `deepseek-flash`，温度 0.1（裁判 0）；
  旧模型名 `deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 停用。
- 分块 200/40 字符，检索 K = 5（混合检索候选 10 条），RRF 参数 k = 60。
- 重排序使用 `BAAI/bge-reranker-base` 交叉编码器；若模型不可用，
  `src/models.py` 会自动退化为本地混合重排序（语义 + BM25 融合），
  并在日志中打印所用模式，避免实验中断。
- 想改用完全免费的智谱 GLM：把 `.env` 里的 `DEEPSEEK_BASE_URL` 改为
  `https://open.bigmodel.cn/api/paas/v4`、`DEEPSEEK_MODEL` 改为免费模型名、
  `DEEPSEEK_API_KEY` 填智谱的 Key，代码无需改动。

## LangChain 对接（可选）

若希望改用 LangChain，可将上面的检索逻辑替换为：

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = FAISS.from_texts(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

再配合 `langchain_openai.ChatOpenAI(base_url="https://api.deepseek.com")` 组装链即可。

## 许可

仅用于课程学习与研究，欢迎参考。请勿提交自己的 `.env` 或 API Key。

