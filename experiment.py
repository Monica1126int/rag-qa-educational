# -*- coding: utf-8 -*-
"""兼容入口：等价于 `python run_experiment.py`（使用正式实验的知识库与测试集）。

保留本文件是为了让早期文档中出现的 `python experiment.py` 继续可用。
新用户建议直接使用 `run_experiment.py`，它支持 --kb / --eval / --no-api / --judge-all。

用法:
    python experiment.py                 # 三组配置对比（默认前 5 题做 LLM 裁判）
    python experiment.py --judge-all      # 全部可答题都做裁判
    python experiment.py --no-api         # 只测检索，不调用大模型（不花钱）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--kb", "data/knowledge_v2.txt",
                "--eval", "data/eval_set_v2.json"] + sys.argv[1:]
    main()
