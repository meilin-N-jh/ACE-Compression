#!/usr/bin/env python3
"""
Extract 16P results and count personality types - include ALL types that appeared
Match qwen14b format: count each personality type and get values from first occurrence
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

results_dir = Path(f'{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/results')

MODELS = ['Llama3.1-8B-FP16', 'Llama3.1-8B-BNB-4bit', 'Llama3.1-8B-GPTQ-INT4', 'Llama3.1-8B-AWQ']

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
    personality_first = {}  # {ptype: first occurrence values}

    # Process all non-Avg rows
    for line in lines:
        if line.startswith('|') and 'Avg' not in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 8:
                ptype = parts[1]
                if len(ptype) >= 4 and ptype[:2] in ['EN', 'ES', 'IN', 'IS']:
                    personality_counter[ptype] += 1
                    if ptype not in personality_first:
                        personality_first[ptype] = {
                            'role': parts[2],
                            'extraverted': parts[3],
                            'intuitive': parts[4],
                            'thinking': parts[5],
                            'judging': parts[6],
                            'assertive': parts[7]
                        }

    print(f"\n=== {model} ===")
    print(f"Found {len(personality_counter)} personality types:")
    for ptype, count in personality_counter.most_common():
        data = personality_first[ptype]
        print(f"  {ptype} ({data['role']}): count={count}")

    # Output all personality types (matching qwen14b format)
    for ptype, count in personality_counter.most_common():
        data = personality_first[ptype]
        row = [model, ptype, data['role'], str(count),
               data['extraverted'], data['intuitive'], data['thinking'],
               data['judging'], data['assertive']]
        output_lines.append(','.join(row))

output_file = Path(f'{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/16p_results_summary.csv')
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"\nResults saved to {output_file}")
