# -*- coding: utf-8 -*-
"""只用真实的交叉编码器重跑 hybrid_rerank（C 组），并把结果合并进 results_final.json。

为什么只跑 C 组：A 组（纯向量）与 B 组（混合检索）不依赖重排序模型，
两者的检索排名在加入交叉编码器前后完全一致，重跑只会重复消耗 API 额度。

用法:
    # 有本地代理时（直连不通）
    $env:HTTPS_PROXY="http://127.0.0.1:7897"
    python run_rerank_crossencoder.py
    # 已缓存、想离线跑
    $env:HF_HUB_OFFLINE="1"; python run_rerank_crossencoder.py
"""
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config  # noqa: E402
from pipeline import RAGPipeline  # noqa: E402
from src.generator import Generator  # noqa: E402

KB = BASE / "data" / "knowledge_v2.txt"
EVAL = BASE / "data" / "eval_set_v2.json"
RESULTS = BASE / "results_final.json"


def judge(generator, key, answer):
    prompt = (
        f"请以高校教务知识库为依据，评估下面的回答质量。\n"
        f"参考答案要点：{key}\n"
        f"待评估回答：{answer}\n"
        f"评分标准：1-5 分，考察准确性、忠实度、完整性。只输出一个数字。"
    )
    try:
        r = generator.client.chat.completions.create(
            model=generator.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return float(r.choices[0].message.content.strip())
    except Exception as e:
        print(f"    [裁判失败] {type(e).__name__}: {e}")
        return None


def main():
    config.DATA_FILE = KB
    cases = json.loads(EVAL.read_text(encoding="utf-8"))["cases"]
    old = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"modes": {}}

    print(f"知识库: {KB.name}   测试集: {EVAL.name}（{len(cases)} 题）")
    print(f"重排序模型: {config.RERANK_MODEL}   生成模型: {config.DEEPSEEK_MODEL}\n")

    generator = Generator(config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL)
    pipe = RAGPipeline("hybrid_rerank", generator)
    source = getattr(pipe.reranker, "source", "unknown")
    print(f"实际重排序实现: {source}\n")
    if source != "cross_encoder":
        print("⚠️ 交叉编码器仍未加载成功，本次结果与上次相同，建议检查代理/缓存后重试。")

    answerable = [c for c in cases if "拒答" not in str(c.get("type", ""))]
    abstain = [c for c in cases if "拒答" in str(c.get("type", ""))]

    per_case, hit1, hit3, abs_ok = [], 0, 0, 0
    t0 = time.time()
    for i, c in enumerate(cases, 1):
        hits = pipe.retrieve(c["q"], top_k=config.TOP_K)
        texts = [pipe.chunks[j] for j, _ in hits]
        ranks = [r for r, t in enumerate(texts, 1) if c["key"] in t]
        is_abs = "拒答" in str(c.get("type", ""))
        if is_abs:
            abs_ok += 1 if not ranks else 0
        else:
            hit1 += 1 if 1 in ranks else 0
            hit3 += 1 if any(r <= 3 for r in ranks) else 0
        per_case.append({
            "id": c.get("id", i), "q": c["q"], "key": c["key"], "type": c.get("type", ""),
            "hit@1": None if is_abs else (1 if 1 in ranks else 0),
            "hit@3": None if is_abs else (1 if any(r <= 3 for r in ranks) else 0),
            "hit_rank": ranks[0] if ranks else None,
            "top1_chunk": texts[0][:80], "answer": None, "judge": None,
        })
        flag = "🚫" if is_abs else ("✅" if 1 in ranks else ("🟡" if any(r <= 3 for r in ranks) else "❌"))
        print(f"  {flag} [{i:2d}] rank={ranks[0] if ranks else None}  {c['q'][:34]}")

    recall1 = hit1 / len(answerable)
    recall3 = hit3 / len(answerable)

    print("\n  生成回答并裁判（全部可答题）...")
    scores = []
    for item in per_case:
        c = next(x for x in cases if x.get("id") == item["id"])
        item["answer"] = pipe.answer(c["q"])["answer"]
        s = judge(generator, item["key"], item["answer"])
        item["judge"] = s
        if s is not None:
            scores.append(s)
    judge_avg = sum(scores) / len(scores) if scores else None

    print(f"\n  → C 组（真实交叉编码器）Recall@1 = {recall1:.2f}  Recall@3 = {recall3:.2f}  "
          f"裁判分 = {judge_avg:.2f}  拒答前提 = {abs_ok/len(abstain):.0%}  耗时 {time.time()-t0:.0f}s")

    old.setdefault("modes", {})["hybrid_rerank"] = {
        "recall@1": round(recall1, 4), "recall@3": round(recall3, 4),
        "n_answerable": len(answerable), "n_abstain": len(abstain),
        "abstain_rate": round(abs_ok / len(abstain), 4) if abstain else None,
        "judge_avg": None if judge_avg is None else round(judge_avg, 3),
        "rerank_source": source, "per_case": per_case,
    }
    RESULTS.write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'配置':<16s} | {'Recall@1':>8s} | {'Recall@3':>8s} | {'裁判分':>7s} | 重排序")
    for mode in ["vector", "hybrid", "hybrid_rerank"]:
        r = old["modes"].get(mode)
        if not r:
            continue
        print(f"{mode:<16s} | {r['recall@1']:>8.2f} | {r['recall@3']:>8.2f} | "
              f"{r['judge_avg']:>7.2f} | {r.get('rerank_source') or '-'}")
    print("=" * 70)
    print(f"结果已写回 {RESULTS.name}")


if __name__ == "__main__":
    main()
