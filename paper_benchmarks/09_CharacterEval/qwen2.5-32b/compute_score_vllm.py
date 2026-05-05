#!/usr/bin/env python3
"""Compute CharacterEval scores (adapted from official for qwen2.5-32b)."""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Compute CharacterEval scores')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model display name (e.g., "Qwen2.5-32B-FP16")')

    args = parser.parse_args()

    CHARACTEREVAL_ROOT = Path(__file__).resolve().parents[1]
    model_results_dir = CHARACTEREVAL_ROOT / "qwen2.5-32b" / "results" / args.model_name
    input_file = model_results_dir / f"{args.model_name}_evaluation.jsonl"
    output_file = model_results_dir / f"{args.model_name}_scores.txt"

    print(f"加载评测结果: {input_file}")
    with open(input_file, "r", encoding='utf-8') as f:
        records = json.load(f)

    print(f"共 {len(records)} 条评测记录")

    score_dict = defaultdict(list)
    for record in records:
        metric_en = record['metric_en']
        score = record[metric_en]
        score_dict[metric_en].append(score)

    print("\n" + "=" * 80)
    print(f"CharacterEval 评分结果 - {args.model_name}")
    print("=" * 80)

    results = []
    for metric in sorted(score_dict.keys()):
        scores = score_dict[metric]
        avg_score = sum(scores) / len(scores)
        results.append((metric, avg_score, len(scores)))
        print(f"{metric}: {avg_score:.4f} (n={len(scores)})")

    total_avg = sum([r[1] for r in results]) / len(results)
    print("=" * 80)
    print(f"总分平均: {total_avg:.4f}")
    print("=" * 80)

    print(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CharacterEval 评分结果 - {args.model_name}\n")
        f.write("=" * 80 + "\n\n")

        for metric, avg_score, count in results:
            f.write(f"{metric}: {avg_score:.4f} (n={count})\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"总分平均: {total_avg:.4f}\n")
        f.write("=" * 80 + "\n")

    print("分数计算完成！")


if __name__ == '__main__':
    main()
