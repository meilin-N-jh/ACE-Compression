import json
import argparse
import os

def most_common_element(lst):
    element_freq = {}
    for item in lst:
        element_freq[item] = element_freq.get(item, 0) + 1
    most_common = max(element_freq, key=element_freq.get)
    return most_common


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--try_times", type=int, default=5)
    args = parser.parse_args()

    # Support both llama70B/results and results paths
    if args.input_path == "":
        input_dir = "./results"
    else:
        input_dir = args.input_path

    files = os.listdir(input_dir)
    acc_per_task = {}
    cnt_per_task = {}

    acc_per_ability = {}
    cnt_per_ability = {}

    for file in files:
        with open(f"{input_dir}/{file}", "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

        # Detect format: vLLM format has 'question_idx', official format has 'idx'
        is_vllm_format = 'question_idx' in data[0] if data else False

        if is_vllm_format:
            # vLLM format processing
            answers = ["" for _ in range(len(data) // args.try_times)]
            preds = [[] for _ in range(len(data) // args.try_times)]
            abilities = [""]

            for d in data:
                qid = d['question_idx']

                # Extract answer from model response
                extracted = extract_answer(d['model_response'])
                # Use mapped_prediction if available, otherwise use extracted
                pred = d.get('mapped_prediction', extracted)

                preds[qid].append(pred)
                if answers[qid] == "":
                    answers[qid] = d['answer']

                if len(abilities) <= qid:
                    abilities.extend([""] * (qid + 1 - len(abilities)))
                # Get ability if available (may not be in vLLM format)
                if 'data' in d and '能力\nABILITY' in d['data']:
                    abilities[qid] = d['data']['能力\nABILITY']
        else:
            # Official format processing
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
            # Extract task name from filename
            # Handle both formats: "TaskName_Model_results.jsonl" or "TaskName_Model_results.jsonl"
            task = file.split("_")[0]
            ability = abilities[i] if i < len(abilities) else "Unknown"

            cnt_per_task[task] = cnt_per_task.get(task, 0) + 1
            cnt_per_ability[ability] = cnt_per_ability.get(ability, 0) + 1


            if answers[i] == most_common_element(preds[i]):
                acc_per_task[task] = acc_per_task.get(task, 0) + 1
                acc_per_ability[ability] = acc_per_ability.get(ability, 0) + 1

    for task in acc_per_task.keys():
        acc_per_task[task] /= cnt_per_task[task]

    for ability in acc_per_ability.keys():
        acc_per_ability[ability] /= cnt_per_ability[ability]

    results = {
        "tasks" : acc_per_task,
        "abilities" : acc_per_ability
    }

    # Save results to input directory
    output_file = os.path.join(input_dir, "results.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # Print summary
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print("\nTask Accuracy:")
    for task, acc in sorted(acc_per_task.items()):
        print(f"  {task}: {acc:.4f}")

    print("\nAbility Accuracy:")
    for ability, acc in sorted(acc_per_ability.items()):
        print(f"  {ability}: {acc:.4f}")

    avg_task = sum(acc_per_task.values()) / len(acc_per_task) if acc_per_task else 0
    print(f"\nAverage Task Accuracy: {avg_task:.4f}")
    print(f"\nResults saved to: {output_file}")
    print("="*80 + "\n")
            