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
from pathlib import Path

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
              'Business Iniatives', 'Sales', 'Marketing/Advertising', 'Finance', 'Accounting',
              'Human Resources', 'Office Work', 'Management/Administration', 'Public Speaking',
              'Politics', 'Law']
}

MODELS = ['Qwen2.5-14B-FP16', 'Qwen2.5-14B-BNB-4bit', 'Qwen2.5-14B-GPTQ-INT4', 'Qwen2.5-14B-AWQ', 'Qwen2.5-14B-Trim',
          'Qwen2.5-14B-2:4-Sparse', 'Qwen2.5-14B-Unstructured-Sparse']

results_dir = Path(f'{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/results')

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
    print(f"  Found {len(md_files)} .md files")

    for md_file in md_files:
        # 从文件名提取问卷名称
        # 文件名格式: Qwen2.5-14B-FP16-BFI.md
        stem = md_file.stem
        # 移除模型前缀，找到最后一个 '-' 后的部分
        # 例如: Qwen2.5-14B-FP16-BFI -> BFI
        parts = stem.split('-')
        # 问卷名称是最后一部分
        questionnaire = parts[-1]

        if questionnaire not in QUESTIONNAIRE_ORDER:
            # 尝试其他可能的格式
            # 例如: Qwen-7B-Chat-BFI -> BFI
            for q_name in QUESTIONNAIRE_ORDER.keys():
                if stem.endswith(q_name):
                    questionnaire = q_name
                    break
            else:
                print(f"  Warning: Unknown questionnaire {questionnaire} in {md_file}")
                continue

        # 读取 .md 文件
        with open(md_file, 'r') as f:
            content = f.read()

        # 提取表格数据 - 逐行匹配
        lines = content.split('\n')
        questionnaire_results = {}

        for line in lines:
            # 匹配包含数据的表格行
            if line.startswith('|') and 'pm' in line and '|' in line:
                parts = line.split('|')
                if len(parts) >= 3:
                    # 提取类别名称（第一列）
                    category = parts[1].strip()

                    # 提取数值部分（第二列）
                    value_str = parts[2].strip()

                    # 使用正则提取均值和标准差
                    # 格式: 3.1 $\pm$ 0.1 或 -10.5 $\pm$ 3.6
                    match = re.search(r'(-?[\d.]+)\s+.*?pm.*?\s+([\d.]+)', value_str, re.IGNORECASE)
                    if match:
                        mean = match.group(1)
                        std = match.group(2)
                        questionnaire_results[category] = f"{mean} ± {std}"

        if questionnaire_results:
            all_results[model][questionnaire] = questionnaire_results
            print(f"  {questionnaire}: {len(questionnaire_results)} dimensions")

print("\nSummary:")
for model, q_results in all_results.items():
    print(f"  {model}: {len(q_results)} questionnaires")

# 构建 CSV 数据
header = ['Model']
subheader = ['']
for q in QUESTIONNAIRE_ORDER:
    dimensions = QUESTIONNAIRE_ORDER[q]
    header.extend([q] * len(dimensions))
    subheader.extend(dimensions)

# 构建数据行
data_rows = []
for model in MODELS:
    if model not in all_results:
        print(f"Warning: No results for {model}")
        continue

    row = [model]
    for q in QUESTIONNAIRE_ORDER:
        dimensions = QUESTIONNAIRE_ORDER[q]
        if q not in all_results[model]:
            row.extend(['N/A'] * len(dimensions))
        else:
            q_results = all_results[model][q]
            for dim in dimensions:
                if dim in q_results:
                    row.append(q_results[dim])
                else:
                    row.append('N/A')

    data_rows.append(row)

# 生成 CSV
output_file = Path('qwen25_14b_results_pivot.csv')

with open(output_file, 'w') as f:
    f.write(','.join(header) + '\n')
    f.write(','.join(subheader) + '\n')
    for row in data_rows:
        f.write(','.join(row) + '\n')

print(f"\nResults saved to {output_file.absolute()}")
print(f"Total models: {len(data_rows)}")
