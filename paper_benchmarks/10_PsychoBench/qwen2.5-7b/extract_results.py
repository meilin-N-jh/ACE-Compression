#!/usr/bin/env python3
"""
Extract PsychoBench results from .md files and generate pivot CSV
"""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())

import re
import pandas as pd
from pathlib import Path
from collections import Counter

# 定义问卷顺序和维度
QUESTIONNAIRE_ORDER = {
    'BFI': ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness'],
    'EPQ-R': ['Extraversion', 'Pschoticism', 'Neuroticism', 'Lying'],
    'DTDD': ['Machiavellianism', 'Psychopathy', 'Narcissism'],
    'BSRI': ['Masculine', 'Feminine'],
    'ECR-R': ['Attachment-related Anxiety', 'Attachment-related Avoidance'],
    'EIS': ['Overall'],
    'Empathy': ['Overall'],
    'GSE': ['Overall'],
    'ICB': ['Overall'],
    'LMS': ['Factor rich', 'Factor motivator', 'Factor important'],
    'LOT-R': ['Overall'],
    'WLEIS': ['SEA', 'OEA', 'UOE', 'ROE'],
    'CABIN': ['Mechanics/Electronics', 'Construction/WoodWork', 'Transportation/Machine Operation',
              'Physical/Manual Labor', 'Protective Service', 'Agriculture', 'Nature/Outdoors',
              'Animal Service', 'Athletics', 'Engineering', 'Physical Science', 'Life Science',
              'Medical Science', 'Social Science', 'Humanities', 'Mathematics/Statistics',
              'Information Technology', 'Visual Arts', 'Applied Arts and Design', 'Performing Arts',
              'Music', 'Writing', 'Media', 'Culinary Art', 'Teaching/Education', 'Social Service',
              'Health Care Service', 'Religious Activities', 'Personal Service', 'Professional Advising',
              'Business Iniities', 'Sales', 'Marketing/Advertising', 'Finance', 'Accounting',
              'Human Resources', 'Office Work', 'Management/Administration', 'Public Speaking',
              'Politics', 'Law']
}

MODELS = ['Qwen2.5-7B-FP16', 'Qwen2.5-7B-BNB-4bit', 'Qwen2.5-7B-GPTQ-INT4', 'Qwen2.5-7B-AWQ',
          'Qwen2.5-7B-2:4-Sparse', 'Qwen2.5-7B-Unstructured-Sparse']

results_dir = Path(f'{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/results')

# 存储所有结果
all_results = {}

for model in MODELS:
    model_dir = results_dir / model
    if not model_dir.exists():
        print(f"Warning: {model_dir} does not exist")
        continue

    print(f"Processing {model}...")
    all_results[model] = {}

    # 查找所有 .md 文件
    md_files = list(model_dir.glob('*.md'))

    for md_file in md_files:
        stem = md_file.stem
        parts = stem.split('-')
        questionnaire = parts[-1]

        # Handle 16P separately
        if questionnaire == '16P':
            continue

        if questionnaire not in QUESTIONNAIRE_ORDER:
            for q_name in QUESTIONNAIRE_ORDER.keys():
                if stem.endswith(q_name):
                    questionnaire = q_name
                    break
            else:
                continue

        with open(md_file, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        questionnaire_results = {}

        for line in lines:
            if line.startswith('|') and 'pm' in line and '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    category = parts[1].strip()
                    value_str = parts[2].strip()
                    match = re.search(r'(-?[\d.]+)\s+.*?pm.*?\s+([\d.]+)', value_str, re.IGNORECASE)
                    if match:
                        mean = match.group(1)
                        std = match.group(2)
                        questionnaire_results[category] = f"{mean} ± {std}"

        if questionnaire_results:
            all_results[model][questionnaire] = questionnaire_results

# Generate pivot CSV
print("\nGenerating pivot CSV...")

header_rows = ['', '']
header_rows[0] = 'Model'
header_rows[1] = ''

for questionnaire, dimensions in QUESTIONNAIRE_ORDER.items():
    for dim in dimensions:
        header_rows[0] += f',{questionnaire}'
        header_rows[1] += f',{dim}'

output_file = results_dir.parent / 'qwen25_7b_results_pivot.csv'
with open(output_file, 'w') as f:
    f.write(header_rows[0] + '\n')
    f.write(header_rows[1] + '\n')

    for model in MODELS:
        if model not in all_results:
            continue

        row = model
        model_results = all_results[model]

        for questionnaire, dimensions in QUESTIONNAIRE_ORDER.items():
            questionnaire_results = model_results.get(questionnaire, {})

            for dim in dimensions:
                value = questionnaire_results.get(dim, 'N/A')
                row += f',{value}'

        f.write(row + '\n')

print(f"Results saved to {output_file}")

# Extract 16P results - ALL personality types
print("\nExtracting 16P results (all types)...")
p16_results = []

for model in MODELS:
    model_dir = results_dir / model
    if not model_dir.exists():
        continue

    p16_file = model_dir / f'{model}-16P.md'
    if not p16_file.exists():
        continue

    with open(p16_file, 'r') as f:
        content = f.read()

    # Collect all personality types and their scores
    personality_counts = Counter()
    personality_scores = {}

    for line in content.split('\n'):
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        # Skip header and empty lines
        if len(parts) < 8 or 'Personality' in line or parts[1] == '' or parts[1] == '0' or parts[1] == '1':
            continue

        # Check if this is a data row (starts with | and has numbers)
        try:
            if parts[1].isdigit():
                ptype = parts[2].strip()
                role = parts[3].strip()
                ext = int(parts[4].strip())
                intu = int(parts[5].strip())
                think = int(parts[6].strip())
                jud = int(parts[7].strip())
                ass = int(parts[8].strip())

                personality_counts[(ptype, role)] = personality_counts.get((ptype, role), 0) + 1

                if (ptype, role) not in personality_scores:
                    personality_scores[(ptype, role)] = {'ext': [], 'intu': [], 'think': [], 'jud': [], 'ass': []}

                personality_scores[(ptype, role)]['ext'].append(ext)
                personality_scores[(ptype, role)]['intu'].append(intu)
                personality_scores[(ptype, role)]['think'].append(think)
                personality_scores[(ptype, role)]['jud'].append(jud)
                personality_scores[(ptype, role)]['ass'].append(ass)
        except (ValueError, IndexError):
            continue

    # Add each unique personality type as a row
    for (ptype, role), count in personality_counts.items():
        scores = personality_scores[(ptype, role)]
        p16_results.append({
            'Model': model,
            'Personality Type': ptype,
            'Role': role,
            'Count': count,
            'Extraverted': round(sum(scores['ext']) / len(scores['ext']), 1),
            'Intuitive': round(sum(scores['intu']) / len(scores['intu']), 1),
            'Thinking': round(sum(scores['think']) / len(scores['think']), 1),
            'Judging': round(sum(scores['jud']) / len(scores['jud']), 1),
            'Assertive': round(sum(scores['ass']) / len(scores['ass']), 1)
        })

if p16_results:
    df_16p = pd.DataFrame(p16_results)
    df_16p.to_csv(results_dir.parent / '16p_results_summary.csv', index=False)
    print(f"16P results saved to {results_dir.parent / '16p_results_summary.csv'}")
    print(f"Total {len(p16_results)} personality types extracted")

print("\nDone!")
