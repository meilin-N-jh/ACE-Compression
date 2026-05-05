import os
import json
import csv
import re
import sys
import math
from statistics import mean, stdev

# Configuration
RESULTS_DIR = "paper_benchmarks/10_PsychoBench/benchmark/results"
QUESTIONNAIRES_FILE = "paper_benchmarks/10_PsychoBench/benchmark/questionnaires.json"

# All questionnaires
SELECTED_TESTS = ["BFI", "16P", "EPQ-R", "DTDD", "BSRI", "ECR-R", "EIS", "Empathy", "GSE", "ICB", "LMS", "LOT-R", "WLEIS", "CABIN"]

MODELS_LLAMA = [
    ("FP16", "paper_benchmarks/10_PsychoBench/benchmark/results/Llama-70B-FP16"),
    ("4bit", "paper_benchmarks/10_PsychoBench/benchmark/results/Llama-70B-4bit"),
    ("GPTQ-INT4", "paper_benchmarks/10_PsychoBench/benchmark/results/Llama-70B-GPTQ-INT4"),
    ("GPTQ-INT8", "paper_benchmarks/10_PsychoBench/benchmark/results/Llama-70B-GPTQ-INT8"),
    ("AWQ", "paper_benchmarks/10_PsychoBench/benchmark/results/Llama-70B-AWQ"),
]

MODELS_QWEN = [
    ("Qwen-7B-Chat", "paper_benchmarks/10_PsychoBench/benchmark/results/Qwen-7B-Chat"),
    ("Qwen-7B-Chat-4bit", "paper_benchmarks/10_PsychoBench/benchmark/results/Qwen-7B-Chat-4bit"),
    ("Qwen-7B-Chat-8bit", "paper_benchmarks/10_PsychoBench/benchmark/results/Qwen-7B-Chat-8bit"),
    ("Qwen-7B-Chat-GGUF", "paper_benchmarks/10_PsychoBench/benchmark/results/Qwen-7B-Chat-GGUF-Q4_K_M"),
    ("Qwen-7B-Int4", "paper_benchmarks/10_PsychoBench/benchmark/results/Qwen-7B-Int4"),
]

def load_questionnaires():
    with open(QUESTIONNAIRES_FILE, 'r') as f:
        return json.load(f)

def get_human_norms(questionnaire):
    norms = {}
    if not questionnaire.get("categories"): return norms
    for cat in questionnaire["categories"]:
        cat_name = cat["cat_name"]
        if cat.get("crowd"):
            crowd_data = cat["crowd"][0] 
            norms[cat_name] = {
                "mean": crowd_data["mean"],
                "std": crowd_data["std"],
                "n": crowd_data["n"]
            }
    return norms

def parse_csv_results(file_path, questionnaire):
    if not os.path.exists(file_path):
        return None
    scores_by_cat = {cat["cat_name"]: [] for cat in questionnaire["categories"]}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        order_indices = [i for i, col in enumerate(header) if col.startswith("order")]
        test_data_list = []
        for i in range(len(order_indices)):
            start = order_indices[i] + 1
            end = order_indices[i+1] - 1 if order_indices[i] != order_indices[-1] else len(header)
            for column_index in range(start, end):
                column_data = {}
                f.seek(0)
                next(reader)
                valid_col = True
                for row in reader:
                    q_idx_str = row[start-1]
                    ans_str = row[column_index]
                    try:
                        q_idx = int(q_idx_str)
                        ans_match = re.search(r'\d+', ans_str)
                        if ans_match:
                            val = int(ans_match.group())
                            if q_idx in questionnaire.get("reverse", []):
                                val = questionnaire["scale"] - val
                            column_data[q_idx] = val
                        else:
                            valid_col = False
                    except:
                        valid_col = False
                if valid_col and column_data:
                    test_data_list.append(column_data)

    final_results = {}
    for cat in questionnaire["categories"]:
        cat_scores_per_run = []
        for data in test_data_list:
            scores = []
            for key in data:
                if key in cat["cat_questions"]:
                    scores.append(data[key])
            if scores:
                if questionnaire.get("compute_mode") == "SUM":
                    cat_scores_per_run.append(sum(scores))
                else:
                    cat_scores_per_run.append(mean(scores))
        if len(cat_scores_per_run) > 0:
            final_results[cat["cat_name"]] = {
                "mean": mean(cat_scores_per_run),
                "std": stdev(cat_scores_per_run) if len(cat_scores_per_run) > 1 else 0.0
            }
    return final_results

def find_result_file(model_dir, test_name):
    if not os.path.exists(model_dir): return None
    for f in os.listdir(model_dir):
        if f.endswith(f"{test_name}.csv"):
            return os.path.join(model_dir, f)
    return None

def analyze_models(models):
    q_data = load_questionnaires()
    all_stats = {}
    
    for model_name, model_dir in models:
        all_stats[model_name] = {}
        for test in SELECTED_TESTS:
            q_config = next((item for item in q_data if item["name"] == test), None)
            if not q_config: continue
            
            csv_file = find_result_file(model_dir, test)
            if csv_file:
                stats = parse_csv_results(csv_file, q_config)
                if stats:
                    all_stats[model_name][test] = stats
            else:
                 all_stats[model_name][test] = None
    return all_stats, q_data

def print_result_json(stats):
    # Helper to convert stats to json string for me to read efficiently
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    print("BEGIN_LLAMA_STATS")
    llama_stats, _ = analyze_models(MODELS_LLAMA)
    print_result_json(llama_stats)
    print("END_LLAMA_STATS")
    
    print("BEGIN_QWEN_STATS")
    qwen_stats, _ = analyze_models(MODELS_QWEN)
    print_result_json(qwen_stats)
    print("END_QWEN_STATS")
