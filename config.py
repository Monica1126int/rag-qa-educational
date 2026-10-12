"""全局配置：API Key、模型名、数据路径与检索超参。"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 必须在导入 transformers / sentence-transformers 之前设置网络与 HuggingFace 环境变量。
# 三条可选路线（在 .env 里选一条即可，互不影响代码）：
#   a) 能直连 HuggingFace：HF_ENDPOINT=https://huggingface.co
#   b) 走国内镜像（默认）：HF_ENDPOINT=https://hf-mirror.com
#   c) 有本地代理且直连不通：把 HTTP_PROXY / HTTPS_PROXY 写进 .env，
#      例如 HTTPS_PROXY=http://127.0.0.1:7897（Clash 默认端口）
# 模型下载完成后设置 HF_HUB_OFFLINE=1，即可只用本地缓存、完全不联网。
# ---------------------------------------------------------------------------
load_dotenv()  # 从 .env 读取 API Key 等配置

# 代理：huggingface_hub / requests 会读这两个环境变量
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    _v = os.getenv(_k, "").strip()
    if _v:
        os.environ[_k] = _v

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型缓存位置：优先用用户目录默认缓存（已下载过 bge 模型的机器可直接复用），
# 若不可写（如受限环境）可在 .env 里覆盖 HF_HOME 为工作区内目录。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if os.getenv("HF_HUB_OFFLINE", "").strip() in ("1", "true", "True"):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "knowledge.txt"

# DeepSeek（OpenAI 兼容接口）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# 注意：旧模型名 deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用，
# 现在官方模型为 deepseek-flash（快、便宜）与 deepseek-v4-pro（更强、稍贵）。
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

# 本地模型（HuggingFace，首次运行自动下载）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

# 检索超参
CHUNK_SIZE = 200      # 分块字符数
CHUNK_OVERLAP = 40    # 分块重叠字符数
TOP_K = 5             # 检索返回条数
