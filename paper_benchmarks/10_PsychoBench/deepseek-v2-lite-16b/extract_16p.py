#!/usr/bin/env python3
"""
Extract 16P results and count personality types
"""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())

from pathlib import Path
from collections import Counter

results_dir = Path(f'{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/results')

MODELS = ['DeepSeek-V2-Lite-16B-FP16', 'DeepSeek-V2-Lite-16B-BNB-4bit', 'DeepSeek-V2-Lite-16B-AWQ']

# 统计每个人格类型
output_lines = ['Model,Personality Type,Role,Count,Extraverted,Intuitive,Thinking,Judging,Assertive']

for model in MODELS:
    md_file = results_dir / model / f'{model}-16P.md'
    if not md_file.exists():
        print(f"Warning: {md_file} does not exist")
        continue

    with open(md_file, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    personality_counter = Counter()

    for line in lines:
        if line.startswith('|') and 'Avg' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 3:
                # 检查是否是有效的人格类型行（包含 MBTI 类型如 ESTP, ENFJ 等）
                ptype = parts[1]
                if len(ptype) >= 4 and ptype[:2] in ['EN', 'ES', 'IN', 'IS']:  # MBTI 前缀
                    personality_counter[ptype] += 1

    print(f"\n=== {model} ===")
    print("人格类型出现次数:")
    for ptype, count in personality_counter.most_common():
        print(f"  {ptype}: {count}")

    # 提取 Avg 行作为汇总
    for line in lines:
        if line.startswith('|') and 'Avg' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 5:
                row = [model, parts[1], parts[2], str(personality_counter.get(parts[1], 1)), parts[3], parts[4], parts[5], parts[6], parts[7]]
                output_lines.append(','.join(row))

output_file = Path(f'{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/16p_results_summary.csv')
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"\nResults saved to {output_file}")
