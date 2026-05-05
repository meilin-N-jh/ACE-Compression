from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())

import json
import argparse
import os


def most_common_element(lst):
    element_freq = {}
    for item in lst:
        element_freq[item] = element_freq.get(item, 0) + 1
    most_common = max(element_freq, key=element_freq.get)
    return most_common


def load_task_abilities(data_dir):
    """Load task-to-ability mapping from original data files.

    Returns:
        tuple: (task_to_full_ability, task_to_category)
    """
    task_to_full = {}
    task_to_category = {}

    for file in os.listdir(data_dir):
        if not file.endswith('.jsonl'):
            continue
        task_name = file.replace('.jsonl', '')
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                data = json.loads(first_line)
                ability = data.get('能力\nABILITY', 'Unknown')
                task_to_full[task_name] = ability
                category = ability.split(':')[0].strip() if ':' in ability else ability
                task_to_category[task_name] = category

    return task_to_full, task_to_category


def extract_answer(text):
    if "[[A]]" in text:
        return "A"
    elif "[[B]]" in text:
        return "B"
    elif "[[C]]" in text:
        return "C"
    elif "[[D]]" in text:
        return "D"
    elif "[A]" in text:
        return "A"
    elif "[B]" in text:
        return "B"
    elif "[C]" in text:
        return "C"
    elif "[D]" in text:
        return "D"
    else:
        for i in range(len(text) - 1, -1, -1):
            if text[i] == 'A':
                return "A"
            elif text[i] == 'B':
                return "B"
            elif text[i] == 'C':
                return "C"
            elif text[i] == 'D':
                return "D"
    return "A"


def compute_ci(p, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    se = (p * (1 - p) / n) ** 0.5
    low = max(0.0, p - z * se)
    high = min(1.0, p + z * se)
    return [low, high]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="results",
                        help="Input directory path (default: results)")
    parser.add_argument("--try_times", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default=f"{ARTIFACT_ROOT}/paper_benchmarks/07_ToM-Bench/benchmark/data",
                        help="Path to original data files")
    args = parser.parse_args()

    input_dir = args.input_path

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_dir)
    task_to_full, task_to_category = load_task_abilities(data_dir)
    print(f"Loaded {len(task_to_category)} task-ability mappings from {data_dir}")

    files = os.listdir(input_dir)
    acc_per_task = {}
    cnt_per_task = {}

    acc_per_category = {}
    cnt_per_category = {}

    acc_per_subability = {}
    cnt_per_subability = {}

    for file in files:
        file_path = os.path.join(input_dir, file)
        if not file.endswith('.jsonl'):
            continue

        with open(file_path, "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

        is_vllm_format = 'question_idx' in data[0] if data else False

        if is_vllm_format:
            answers = ["" for _ in range(len(data) // args.try_times)]
            preds = [[] for _ in range(len(data) // args.try_times)]

            for d in data:
                qid = d['question_idx']
                extracted = extract_answer(d['model_response'])
                pred = d.get('mapped_prediction', extracted)
                preds[qid].append(pred)
                if answers[qid] == "":
                    answers[qid] = d['answer']
        else:
            answers = ["" for _ in range(len(data) // args.try_times)]
            preds = [[] for _ in range(len(data) // args.try_times)]
            abilities = ["" for _ in range(len(data) // args.try_times)]
            for d in data:
                preds[d['idx']].append(d['map'][extract_answer(d['output'])])
                if answers[d['idx']] == "":
                    answers[d['idx']] = d['answer']

                if abilities[d['idx']] == "":
                    abilities[d['idx']] = d['data']['能力\nABILITY']

        for i in range(len(data) // args.try_times):
            # Extract task name from filename: {Task}_{Model}_results.jsonl
            # Find the start of model name (_Qwen2.5-14B)
            if "_Qwen2.5-14B" in file:
                task = file.split("_Qwen2.5-14B")[0]
            else:
                task = file.split("_")[0]  # fallback
            category = task_to_category.get(task, "Unknown")
            sub_ability = task_to_full.get(task, "Unknown")

            cnt_per_task[task] = cnt_per_task.get(task, 0) + 1
            cnt_per_category[category] = cnt_per_category.get(category, 0) + 1
            cnt_per_subability[sub_ability] = cnt_per_subability.get(sub_ability, 0) + 1

            if answers[i] == most_common_element(preds[i]):
                acc_per_task[task] = acc_per_task.get(task, 0) + 1
                acc_per_category[category] = acc_per_category.get(category, 0) + 1
                acc_per_subability[sub_ability] = acc_per_subability.get(sub_ability, 0) + 1

    for task in acc_per_task.keys():
        acc_per_task[task] /= cnt_per_task[task]

    for category in acc_per_category.keys():
        acc_per_category[category] /= cnt_per_category[category]

    for sub_ability in acc_per_subability.keys():
        acc_per_subability[sub_ability] /= cnt_per_subability[sub_ability]

    tasks_ci = {k: compute_ci(acc_per_task[k], cnt_per_task[k]) for k in acc_per_task}
    abilities_ci = {k: compute_ci(acc_per_category[k], cnt_per_category[k]) for k in acc_per_category}

    results = {
        "tasks": acc_per_task,
        "abilities": acc_per_category,
        "sub_abilities": acc_per_subability,
        "tasks_ci": tasks_ci,
        "abilities_ci": abilities_ci,
        "tasks_n": cnt_per_task,
        "abilities_n": cnt_per_category
    }

    output_file = os.path.join(input_dir, "results.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print("\nTask Accuracy:")
    for task, acc in sorted(acc_per_task.items()):
        print(f"  {task}: {acc:.4f}")

    print("\n" + "-" * 80)
    print("6 Major Ability Categories:")
    for category, acc in sorted(acc_per_category.items()):
        print(f"  {category}: {acc:.4f}")

    print("\n" + "-" * 80)
    print("Sub-Abilities (细分能力):")
    for sub_ability, acc in sorted(acc_per_subability.items()):
        print(f"  {sub_ability}: {acc:.4f}")

    avg_task = sum(acc_per_task.values()) / len(acc_per_task) if acc_per_task else 0
    avg_category = sum(acc_per_category.values()) / len(acc_per_category) if acc_per_category else 0
    print(f"\nAverage Task Accuracy: {avg_task:.4f}")
    print(f"Average Category Accuracy: {avg_category:.4f}")
