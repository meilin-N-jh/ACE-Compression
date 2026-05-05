#!/usr/bin/env python3
"""
PsychoBench Results Analyzer
自动解析所有模型的CSV数据，计算Mean±SD，生成汇总表格
"""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())


import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import csv

# 配置
RESULTS_DIR = f"{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/results"
OUTPUT_DIR = f"{ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark/llama70B/results/analysis"
MODELS = {
    "FP16": "Llama-70B-FP16",
    "4bit": "Llama-70B-4bit",
    "GPTQ-Int4": "Llama-70B-GPTQ-INT4",
    "GPTQ-Int8": "Llama-70B-GPTQ-INT8",
    "AWQ": "Llama-70B-AWQ",
}

# 跳过16P问卷
SKIP_QUESTIONNAIRES = ["16P"]

# 问卷列表
QUESTIONNAIRES = [
    "BFI", "DTDD", "EPQ-R", "ECR-R", "CABIN", "GSE", "LMS",
    "BSRI", "ICB", "LOT-R", "Empathy", "EIS", "WLEIS"
]

# BFI问卷计分规则（根据官方文献）
# 格式: (题号, 是否反向计分)
BFI_SCORING = {
    "E": [(1, False), (6, True), (11, True), (16, False), (21, True), (26, False), (31, True), (36, False)],  # Extraversion
    "A": [(2, True), (7, False), (12, True), (17, False), (22, False), (27, True), (32, False), (37, True), (42, False)],  # Agreeableness
    "C": [(3, True), (8, True), (13, False), (18, True), (23, True), (28, False), (33, False), (38, False), (43, True)],  # Conscientiousness
    "N": [(4, False), (9, True), (14, False), (19, False), (24, True), (29, False), (34, True), (39, False)],  # Neuroticism
    "O": [(5, False), (10, False), (15, False), (20, True), (25, False), (30, False), (35, True), (40, True), (41, True), (44, False)]  # Openness
}

# DTDD问卷计分规则（Dark Triad Dirty Dozen）
DTDD_SCORING = {
    "Mach": [1, 2, 3, 4],  # Machiavellianism
    "Psych": [5, 6, 7, 8],  # Psychopathy
    "Narc": [9, 10, 11, 12]  # Narcissism
}

def read_csv_data(csv_path):
    """
    读取PsychoBench生成的CSV文件
    返回: 问题得分矩阵 (n_questions × n_tests)
    """
    try:
        df = pd.read_csv(csv_path)
        # 找到所有测试列（shuffle0-testX 和 shuffle1-testX）
        test_cols = [col for col in df.columns if 'shuffle' in col and 'test' in col]

        if len(test_cols) == 0:
            print(f"  警告: {csv_path} 中没有找到测试列")
            return None

        # 提取得分数据（跳过第一行的prompt列）
        scores = df.iloc[1:, df.columns.get_indexer(test_cols)].values

        # 转换为数值类型
        scores = pd.to_numeric(scores.flatten(), errors='coerce').reshape(scores.shape)

        return scores
    except Exception as e:
        print(f"  错误读取 {csv_path}: {e}")
        return None


def calculate_bfi_scores(scores):
    """
    计算BFI各维度得分
    scores: numpy array (n_questions × n_tests)
    """
    if scores is None or scores.shape[0] < 44:
        return None

    # 题号从1开始，需要调整索引
    results = {}

    for dimension, questions in BFI_SCORING.items():
        dimension_scores = []

        for q, is_reverse in questions:
            q_idx = q - 1  # 转换为0-based索引

            # 获取该题在所有测试中的得分
            q_scores = scores[q_idx, :]

            # 反向计分：1->5, 2->4, 3->3, 4->2, 5->1
            if is_reverse:
                q_scores = 6 - q_scores

            dimension_scores.append(q_scores)

        # 计算该维度的平均分（每个测试）
        dimension_mean = np.mean(dimension_scores, axis=0)
        results[dimension] = dimension_mean

    return results


def calculate_dtdd_scores(scores):
    """
    计算DTDD各维度得分
    scores: numpy array (n_questions × n_tests)
    """
    if scores is None or scores.shape[0] < 12:
        return None

    results = {}

    for dimension, questions in DTDD_SCORING.items():
        dimension_scores = []

        for q in questions:
            q_idx = q - 1  # 转换为0-based索引
            q_scores = scores[q_idx, :]
            dimension_scores.append(q_scores)

        # 计算该维度的平均分
        dimension_mean = np.mean(dimension_scores, axis=0)
        results[dimension] = dimension_mean

    return results


def calculate_total_score(scores):
    """
    计算总分（适用于GSE, Empathy等）
    """
    if scores is None:
        return None

    # 计算每个测试的总分
    total_scores = np.nanmean(scores, axis=0)
    return total_scores


def extract_score_from_md(md_path):
    """
    从MD文件中提取已经计算好的分数
    作为备用方案
    """
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找包含mean和std的行
        import re
        pattern = r'\|\s*(\w+)\s*\|\s*([\d.]+)\s*\$\?pm\$?\s*\|\s*([\d.]+)\s*\|'
        matches = re.findall(pattern, content.replace('$\\pm$', '±'))

        results = {}
        for match in matches:
            dim_name, mean, std = match
            results[dim_name] = {'mean': float(mean), 'std': float(std)}

        return results
    except Exception as e:
        print(f"  错误读取 {md_path}: {e}")
        return None


def process_model(model_name, model_dir):
    """
    处理单个模型的所有问卷
    """
    print(f"\n处理模型: {model_name}")

    model_results = {
        "model": model_name,
        "questionnaires": {}
    }

    # 遍历所有问卷文件
    for questionnaire in QUESTIONNAIRES:
        if any(skip in questionnaire for skip in SKIP_QUESTIONNAIRES):
            continue

        csv_file = os.path.join(model_dir, f"{model_name}-{questionnaire}.csv")
        md_file = os.path.join(model_dir, f"{model_name}-{questionnaire}.md")

        if not os.path.exists(csv_file):
            print(f"  跳过 {questionnaire}: 文件不存在")
            continue

        print(f"  处理 {questionnaire}...")

        # 读取CSV数据
        scores = read_csv_data(csv_file)

        # 根据问卷类型计算得分
        if questionnaire == "BFI":
            dim_scores = calculate_bfi_scores(scores)
            if dim_scores:
                model_results["questionnaires"]["BFI"] = {}
                for dim, scores_array in dim_scores.items():
                    model_results["questionnaires"]["BFI"][dim] = {
                        "mean": float(np.nanmean(scores_array)),
                        "std": float(np.nanstd(scores_array, ddof=1))
                    }

        elif questionnaire == "DTDD":
            dim_scores = calculate_dtdd_scores(scores)
            if dim_scores:
                model_results["questionnaires"]["DTDD"] = {}
                for dim, scores_array in dim_scores.items():
                    model_results["questionnaires"]["DTDD"][dim] = {
                        "mean": float(np.nanmean(scores_array)),
                        "std": float(np.nanstd(scores_array, ddof=1))
                    }

        elif questionnaire in ["GSE", "Empathy", "EIS", "LOT-R"]:
            total_scores = calculate_total_score(scores)
            if total_scores is not None:
                model_results["questionnaires"][questionnaire] = {
                    "mean": float(np.nanmean(total_scores)),
                    "std": float(np.nanstd(total_scores, ddof=1))
                }

        # 对于复杂的问卷，暂时从MD文件提取
        if questionnaire in ["EPQ-R", "ECR-R", "CABIN", "LMS", "BSRI", "ICB", "WLEIS"]:
            md_results = extract_score_from_md(md_file)
            if md_results:
                model_results["questionnaires"][questionnaire] = md_results

    return model_results


def generate_summary_table(all_results):
    """
    生成汇总表格（Markdown格式）
    """
    print("\n生成汇总表格...")

    # BFI汇总表
    bfi_table = []
    bfi_table.append("| Model | BFI-E | BFI-A | BFI-C | BFI-N | BFI-O |")
    bfi_table.append("|:---|:---:|:---:|:---:|:---:|:---:|")

    for model_key, model_name in MODELS.items():
        if model_name not in all_results:
            continue

        model_data = all_results[model_name]
        if "BFI" not in model_data["questionnaires"]:
            continue

        bfi = model_data["questionnaires"]["BFI"]
        row = f"| **{model_key}** |"

        for dim in ["E", "A", "C", "N", "O"]:
            if dim in bfi:
                mean = bfi[dim]["mean"]
                std = bfi[dim]["std"]
                row += f" {mean:.1f}±{std:.1f} |"
            else:
                row += " N/A |"

        bfi_table.append(row)

    # DTDD汇总表
    dtdd_table = []
    dtdd_table.append("| Model | DTDD-Mach | DTDD-Psych | DTDD-Narc |")
    dtdd_table.append("|:---|:---:|:---:|:---:|")

    for model_key, model_name in MODELS.items():
        if model_name not in all_results:
            continue

        model_data = all_results[model_name]
        if "DTDD" not in model_data["questionnaires"]:
            continue

        dtdd = model_data["questionnaires"]["DTDD"]
        row = f"| **{model_key}** |"

        for dim in ["Mach", "Psych", "Narc"]:
            if dim in dtdd:
                mean = dtdd[dim]["mean"]
                std = dtdd[dim]["std"]
                row += f" {mean:.1f}±{std:.1f} |"
            else:
                row += " N/A |"

        dtdd_table.append(row)

    # 其他量表汇总表
    other_table = []
    other_table.append("| Model | GSE | Empathy | EIS | LOT-R |")
    other_table.append("|:---|:---:|:---:|:---:|:---:|")

    for model_key, model_name in MODELS.items():
        if model_name not in all_results:
            continue

        model_data = all_results[model_name]
        row = f"| **{model_key}** |"

        for scale in ["GSE", "Empathy", "EIS", "LOT-R"]:
            if scale in model_data["questionnaires"]:
                scale_data = model_data["questionnaires"][scale]
                if isinstance(scale_data, dict):
                    if "mean" in scale_data:
                        mean = scale_data["mean"]
                        std = scale_data["std"]
                        row += f" {mean:.1f}±{std:.1f} |"
                    else:
                        row += " N/A |"
                else:
                    row += " N/A |"
            else:
                row += " N/A |"

        other_table.append(row)

    return "\n".join(bfi_table), "\n".join(dtdd_table), "\n".join(other_table)


def main():
    """
    主函数
    """
    print("=" * 60)
    print("PsychoBench Results Analyzer")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 处理所有模型
    all_results = {}

    for model_key, model_name in MODELS.items():
        model_dir = os.path.join(RESULTS_DIR, model_name)

        if not os.path.exists(model_dir):
            print(f"警告: 模型目录不存在: {model_dir}")
            continue

        model_results = process_model(model_name, model_dir)
        all_results[model_name] = model_results

    # 保存JSON结果
    json_path = os.path.join(OUTPUT_DIR, "psychobench_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {json_path}")

    # 生成汇总表格
    bfi_table, dtdd_table, other_table = generate_summary_table(all_results)

    # 保存到文件
    summary_path = os.path.join(OUTPUT_DIR, "PSYCHOBENCH_SUMMARY_TABLES.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# PsychoBench Results Summary\n\n")
        f.write("## Table 1: Big Five Inventory (BFI)\n\n")
        f.write(bfi_table + "\n\n")
        f.write("## Table 2: Dark Triad Dirty Dozen (DTDD)\n\n")
        f.write(dtdd_table + "\n\n")
        f.write("## Table 3: Other Scales\n\n")
        f.write(other_table + "\n")

    print(f"汇总表格已保存到: {summary_path}")

    # 打印表格到控制台
    print("\n" + "=" * 60)
    print("BFI Results:")
    print("=" * 60)
    print(bfi_table)

    print("\n" + "=" * 60)
    print("DTDD Results:")
    print("=" * 60)
    print(dtdd_table)

    print("\n" + "=" * 60)
    print("Other Scales:")
    print("=" * 60)
    print(other_table)


if __name__ == "__main__":
    main()
