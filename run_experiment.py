# -*- coding: utf-8 -*-
"""在给定知识库与测试集上做详细实验，输出逐题检索细节与答案。

用法:
    python run_experiment.py                      # 用 data/knowledge.txt + data/eval_set.json
    python run_experiment.py --kb data/knowledge_v2.txt --eval data/eval_set_v2.json
    python run_experiment.py --no-api             # 只做检索评测，不调用大模型（省钱）

输出:
    results.json   逐题命中情况、Top-k 片段、生成答案、裁判分
    屏幕打印逐题表与汇总表
"""
import argparse
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config  # noqa: E402
from pipeline import RAGPipeline  # noqa: E402
from src.generator import Generator  # noqa: E402


def load_eval_set(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"]


def is_abstain(case) -> bool:
    """「无依据拒答」类题不计入 Recall 分母，单独统计。"""
    return "拒答" in str(case.get("type", ""))


def judge(generator, case, answer):
    """LLM 裁判：1-5 分，考察准确性/忠实度/完整性。"""
    prompt = (
        f"请以高校教务知识库为依据，评估下面的回答质量。\n"
        f"参考答案要点：{case['key']}\n"
        f"待评估回答：{answer}\n"
        f"评分标准：1-5 分，考察准确性、忠实度、完整性。只输出一个数字。"
    )
    try:
        resp = generator.client.chat.completions.create(
            model=generator.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return float(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"      [裁判失败] {type(e).__name__}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="data/knowledge.txt")
    ap.add_argument("--eval", default="data/eval_set.json")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--no-api", action="store_true", help="跳过生成与裁判，只测检索")
    ap.add_argument("--judge-all", action="store_true", help="对所有题打分（默认只前 5 题）")
    args = ap.parse_args()

    kb_path = BASE / args.kb
    eval_path = BASE / args.eval
    config.DATA_FILE = kb_path  # 覆盖配置里的知识库路径

    cases = load_eval_set(eval_path)
    generator = None if args.no_api else Generator(
        config.DEEPSEEK_API_KEY, config.DEEPSEEK_BASE_URL, config.DEEPSEEK_MODEL
    )

    print(f"知识库: {kb_path.name}")
    print(f"测试集: {eval_path.name}（{len(cases)} 题）")
    print(f"生成与裁判: {'跳过（--no-api）' if args.no_api else config.DEEPSEEK_MODEL}")
    print()

    all_results = {
        "kb": kb_path.name,
        "eval": eval_path.name,
        "n_cases": len(cases),
        "model": config.DEEPSEEK_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "modes": {},
    }

    for mode in RAGPipeline.MODES:
        print("=" * 78)
        print(f"配置：{mode}")
        print("=" * 78)
        pipe = RAGPipeline(mode, generator or _DummyGenerator())
        rerank_source = pipe.reranker.source if getattr(pipe, "reranker", None) else None

        per_case = []
        n_ans = sum(1 for c in cases if not is_abstain(c))
        n_abs = len(cases) - n_ans
        hit1 = hit3 = abs_ok = 0
        t0 = time.time()

        for i, case in enumerate(cases, 1):
            hits = pipe.retrieve(case["q"], top_k=config.TOP_K)
            texts = [pipe.chunks[j] for j, _ in hits]
            ranks = [r for r, t in enumerate(texts, 1) if case["key"] in t]
            abstain = is_abstain(case)
            h1 = 1 if 1 in ranks else 0
            h3 = 1 if any(r <= 3 for r in ranks) else 0
            # 拒答题：Top-K 里没有出现关键串，才具备「答不了」的前提
            if abstain:
                abs_ok += 1 if not ranks else 0
            else:
                hit1 += h1
                hit3 += h3

            item = {
                "id": case.get("id", i),
                "q": case["q"],
                "key": case["key"],
                "type": case.get("type", ""),
                "hit@1": None if abstain else h1,
                "hit@3": None if abstain else h3,
                "hit_rank": ranks[0] if ranks else None,
                "top1_chunk": texts[0][:80],
                "answer": None,
                "judge": None,
            }

            if generator is not None:
                ctx = "\n".join(texts)
                item["answer"] = generator.generate(case["q"], ctx)

            per_case.append(item)
            if abstain:
                flag = "🚫" if not ranks else "⚠️"
                print(f"  {flag} [{i:2d}] 拒答题 依据{'未命中(应拒答)' if not ranks else '命中关键串'}  {case['q'][:30]}")
            else:
                flag = "✅" if h1 else ("🟡" if h3 else "❌")
                print(f"  {flag} [{i:2d}] rank={item['hit_rank']}  {case['q'][:34]}")

        recall1 = hit1 / n_ans if n_ans else float("nan")
        recall3 = hit3 / n_ans if n_ans else float("nan")
        abstain_rate = abs_ok / n_abs if n_abs else None

        # 裁判打分
        judge_avg = None
        if generator is not None:
            n_judge = len(cases) if args.judge_all else min(5, len(cases))
            scores = []
            print(f"\n  LLM 裁判（前 {n_judge} 题）...")
            for item in per_case[:n_judge]:
                s = judge(generator, {"key": item["key"]}, item["answer"])
                item["judge"] = s
                if s is not None:
                    scores.append(s)
                print(f"    {item['id']:>3} 分={s}  {item['q'][:28]}")
            judge_avg = sum(scores) / len(scores) if scores else None

        print(f"\n  → Recall@1 = {recall1:.2f}   Recall@3 = {recall3:.2f}   "
              f"裁判分 = {'—' if judge_avg is None else f'{judge_avg:.2f}'}   "
              f"拒答题前提成立率 = {'—' if abstain_rate is None else f'{abstain_rate:.0%}'}   "
              f"耗时 {time.time()-t0:.1f}s\n")

        all_results["modes"][mode] = {
            "recall@1": round(recall1, 4),
            "recall@3": round(recall3, 4),
            "n_answerable": n_ans,
            "n_abstain": n_abs,
            "abstain_rate": None if abstain_rate is None else round(abstain_rate, 4),
            "judge_avg": None if judge_avg is None else round(judge_avg, 3),
            "rerank_source": rerank_source,
            "per_case": per_case,
        }

    # 汇总
    print("=" * 78)
    print("汇总对比")
    print("=" * 78)
    print(f"{'配置':<16s} | {'Recall@1':>8s} | {'Recall@3':>8s} | {'裁判分':>7s} | {'拒答前提':>8s} | 重排序")
    for mode, r in all_results["modes"].items():
        j = "—" if r["judge_avg"] is None else f"{r['judge_avg']:.2f}"
        a = "—" if r["abstain_rate"] is None else f"{r['abstain_rate']:.0%}"
        print(f"{mode:<16s} | {r['recall@1']:>8.2f} | {r['recall@3']:>8.2f} | {j:>7s} | {a:>8s} | "
              f"{r['rerank_source'] or '-'}")

    out = BASE / args.out
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已保存: {out}")


class _DummyGenerator:
    """--no-api 模式下占位，保证 pipeline 可构造。"""

    model = "none"

    class _C:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("no-api 模式不应调用生成")

    client = _C()

    def generate(self, query, context):
        return ""


if __name__ == "__main__":
    main()
