#!/usr/bin/env python3
"""
使用官方utils.py进行PsychoBench结果分析
严格按照官方仓库的compute_mode处理SUM和AVG
"""

import sys
import os
import json
import pandas as pd
from utils import convert_data, compute_statistics

def analyze_all_questionnaires():
    """分析所有问卷结果"""

    # 加载问卷配置
    with open('questionnaires.json', 'r') as f:
        questionnaires_data = json.load(f)

    # 创建问卷名称到配置的映射
    questionnaire_configs = {}
    for item in questionnaires_data:
        questionnaire_configs[item['name']] = item

    # 要分析的问卷列表
    questionnaire_names = [
        "BFI", "DTDD", "EPQ-R", "ECR-R", "CABIN",
        "GSE", "LMS", "BSRI", "ICB", "LOT-R",
        "Empathy", "EIS", "WLEIS", "16P"
    ]

    results_dir = "results/Qwen-7B-Chat"

    print("# Qwen-7B-Chat PsychoBench 官方分析结果")
    print("=" * 60)

    total_successful = 0
    total_questionnaires = len(questionnaire_names)

    for q_name in questionnaire_names:
        print(f"\n## {q_name}")

        # 检查配置
        if q_name not in questionnaire_configs:
            print("❌ 问卷配置不存在")
            continue

        questionnaire = questionnaire_configs[q_name]

        # 检查数据文件
        csv_file = os.path.join(results_dir, f"Qwen-7B-Chat-{q_name}.csv")
        if not os.path.exists(csv_file):
            print(f"❌ 数据文件不存在: {csv_file}")
            continue

        try:
            # 转换数据
            test_data = convert_data(questionnaire, csv_file)

            if not test_data:
                print("❌ 没有有效数据")
                continue

            print(f"✅ 成功加载 {len(test_data)} 个测试案例")

            # 计算统计
            test_results = compute_statistics(questionnaire, test_data)

            if not test_results:
                print("❌ 无法计算统计数据")
                continue

            print(f"📊 计算模式: {questionnaire.get('compute_mode', 'AVG')}")

            # 显示各维度结果
            for i, cat in enumerate(questionnaire['categories']):
                if i < len(test_results):
                    mean_val, std_val, n_val = test_results[i]
                    print(f"   - {cat['cat_name']}: {mean_val:.2f} ± {std_val:.2f} (n={n_val})")

            total_successful += 1

        except Exception as e:
            print(f"❌ 分析失败: {e}")
            # 打印详细错误用于调试
            import traceback
            traceback.print_exc()

    print(f"\n" + "=" * 60)
    print(f"## 总结")
    print(f"成功分析: {total_successful}/{total_questionnaires} 个问卷")
    print(f"成功率: {total_successful/total_questionnaires*100:.1f}%")

    if total_successful >= total_questionnaires * 0.8:
        print("🎉 评测基本成功！")
    elif total_successful >= total_questionnaires * 0.5:
        print("⚠️ 评测部分成功")
    else:
        print("❌ 评测成功率较低")

if __name__ == "__main__":
    analyze_all_questionnaires()