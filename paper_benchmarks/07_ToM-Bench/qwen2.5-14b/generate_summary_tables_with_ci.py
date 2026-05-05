#!/usr/bin/env python3
"""Generate separate Task and Ability summary CSVs for ToM-Bench with confidence intervals."""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())

import json
import os
import math
from pathlib import Path
from collections import Counter

MODELS = [
    'Qwen2.5-14B-FP16',
    'Qwen2.5-14B-BNB-4bit',
    'Qwen2.5-14B-GPTQ-INT4',
    'Qwen2.5-14B-AWQ',
    'Qwen2.5-14B-Trim'
]

RESULTS_DIR = Path('results')
DATA_DIR = Path(f'{ARTIFACT_ROOT}/paper_benchmarks/07_ToM-Bench/benchmark/data')

def compute_ci(p, n, z=1.96):
    """Compute confidence interval for proportion using Wilson score interval."""
    if n == 0:
        return 0.0, 0.0
    if p == 0 or p == 1:
        return 0.0, 0.0

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper

# Load task-to-ability mapping
task_to_category = {}
task_to_full_ability = {}

for file in os.listdir(DATA_DIR):
    if not file.endswith('.jsonl'):
        continue
    task_name = file.replace('.jsonl', '')
    file_path = DATA_DIR / file
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        if first_line:
            data = json.loads(first_line)
            ability = data.get('能力\nABILITY', 'Unknown')
            task_to_full_ability[task_name] = ability
            category = ability.split(':')[0].strip() if ':' in ability else ability
            task_to_category[task_name] = category

# Store results for each model
model_task_results = {}  # {model: {task: {'accuracy': acc, 'correct': c, 'total': t}}}
model_ability_results = {}  # {model: {ability: {'accuracy': acc, 'correct': c, 'total': t}}}

for model in MODELS:
    model_task_results[model] = {}
    model_ability_results[model] = {}

    model_files = list(RESULTS_DIR.glob(f'*{model}_results.jsonl'))
    print(f"Processing {model}... ({len(model_files)} files)")

    for result_file in model_files:
        # Extract task name
        task = result_file.stem.replace(f'_{model}_results', '')

        with open(result_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]

        # Process records using majority vote
        answers = []
        preds = [[] for _ in range(len(records) // 5)]  # 5 trials

        for i, record in enumerate(records):
            qid = record['question_idx']
            if len(preds[qid]) < 5:
                preds[qid].append(record['mapped_prediction'])
                if len(answers) <= qid:
                    answers.append(record['answer'])

        # Calculate accuracy using majority vote
        correct = 0
        total = len(preds)
        for i in range(total):
            if preds[i]:
                most_common = Counter(preds[i]).most_common(1)[0][0]
                if most_common == answers[i]:
                    correct += 1

        accuracy = correct / total if total > 0 else 0
        model_task_results[model][task] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

        # Group by ability category
        category = task_to_category.get(task, 'Unknown')
        if category not in model_ability_results[model]:
            model_ability_results[model][category] = {'correct': 0, 'total': 0}
        model_ability_results[model][category]['correct'] += correct
        model_ability_results[model][category]['total'] += total

# Calculate average ability accuracies
for model in MODELS:
    for ability in model_ability_results[model]:
        stats = model_ability_results[model][ability]
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

# Get all unique tasks and abilities (sorted)
all_tasks = sorted(set(task for model_tasks in model_task_results.values() for task in model_tasks))
all_abilities = sorted(set(ability for model_ab in model_ability_results.values() for ability in model_ab))

# Generate Task CSV with CI
task_csv = 'tombench_task_summary_with_ci.csv'
with open(task_csv, 'w', encoding='utf-8') as f:
    # Header
    header = ['Model'] + all_tasks
    f.write(','.join(header) + '\n')

    # Rows
    for model in MODELS:
        row = [model]
        for task in all_tasks:
            if task in model_task_results[model]:
                acc = model_task_results[model][task]['accuracy']
                correct = model_task_results[model][task]['correct']
                total = model_task_results[model][task]['total']
                lower, upper = compute_ci(acc, total)
                # Format: percentage ± CI
                ci_lower_pct = lower * 100
                ci_upper_pct = upper * 100
                ci_margin = (upper - lower) * 100 / 2
                row.append(f"{acc*100:.2f}% ± {ci_margin:.2f}%")
            else:
                row.append('N/A')
        f.write(','.join(row) + '\n')

print(f"\nTask results (with CI) saved to {task_csv}")

# Generate Ability CSV with CI
ability_csv = 'tombench_ability_summary_with_ci.csv'
with open(ability_csv, 'w', encoding='utf-8') as f:
    # Header
    header = ['Model'] + all_abilities
    f.write(','.join(header) + '\n')

    # Rows
    for model in MODELS:
        row = [model]
        for ability in all_abilities:
            if ability in model_ability_results[model]:
                acc = model_ability_results[model][ability]['accuracy']
                correct = model_ability_results[model][ability]['correct']
                total = model_ability_results[model][ability]['total']
                lower, upper = compute_ci(acc, total)
                # Format: percentage ± CI
                ci_margin = (upper - lower) * 100 / 2
                row.append(f"{acc*100:.2f}% ± {ci_margin:.2f}%")
            else:
                row.append('N/A')
        f.write(','.join(row) + '\n')

print(f"Ability results (with CI) saved to {ability_csv}")

# Print summary
print("\n" + "=" * 80)
print("ToM-Bench Results Summary (with 95% CI)")
print("=" * 80)
print("\nTask Averages:")
for model in MODELS:
    if model in model_task_results:
        total_correct = sum(r['correct'] for r in model_task_results[model].values())
        total_questions = sum(r['total'] for r in model_task_results[model].values())
        avg = total_correct / total_questions if total_questions > 0 else 0
        print(f"  {model}: {avg*100:.2f}%")

print("\nAbility Averages:")
for model in MODELS:
    if model in model_ability_results:
        total_correct = sum(r['correct'] for r in model_ability_results[model].values())
        total_questions = sum(r['total'] for r in model_ability_results[model].values())
        avg = total_correct / total_questions if total_questions > 0 else 0
        print(f"  {model}: {avg*100:.2f}%")
