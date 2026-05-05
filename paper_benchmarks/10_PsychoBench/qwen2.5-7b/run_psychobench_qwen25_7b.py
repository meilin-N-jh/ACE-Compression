#!/usr/bin/env python3
"""
Run PsychoBench with Qwen2.5-7B via vLLM API
用法: python run_psychobench_qwen25_7b.py --base-url http://127.0.0.1:8400/v1 --model qwen2.5-7b-fp16 --questionnaire Empathy,BFI,BSRI --shuffle-count 1 --test-count 10
"""

import argparse
import os
import sys

# 添加父目录到路径并切换到PsychoBench根目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

# 添加当前目录到路径以导入generator
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import run_psychobench
from qwen25_7b_vllm_generator import qwen25_14b_vllm_generator


def main():
    parser = argparse.ArgumentParser(description="Run PsychoBench with Qwen2.5-14B via vLLM API")

    # 模型配置
    parser.add_argument('--base-url', required=True, type=str,
                       help='vLLM API base URL (e.g., http://127.0.0.1:8100/v1)')
    parser.add_argument('--model', required=True, type=str,
                       help='Model name (e.g., qwen2.5-14b-fp16)')
    parser.add_argument('--api-key', type=str, default='EMPTY',
                       help='API key (default: EMPTY)')

    # 问卷配置
    parser.add_argument('--questionnaire', required=True, type=str,
                       help='Comma-separated list of questionnaire names')
    parser.add_argument('--shuffle-count', type=int, default=1,
                       help='Number of shuffles (default: 1)')
    parser.add_argument('--test-count', type=int, default=10,
                       help='Number of tests per shuffle (default: 10)')

    # 其他配置
    parser.add_argument('--name-exp', type=str, default=None,
                       help='Experiment name (default: model name)')
    parser.add_argument('--significance-level', type=float, default=0.01,
                       help='Significance level for statistics (default: 0.01)')
    parser.add_argument('--mode', type=str, default='auto',
                       help='Mode (default: auto)')

    args = parser.parse_args()

    # 设置默认实验名称
    if args.name_exp is None:
        args.name_exp = args.model

    print("=" * 60)
    print("PsychoBench: Qwen2.5-14B via vLLM API")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Questionnaires: {args.questionnaire}")
    print(f"Shuffle count: {args.shuffle_count}")
    print(f"Test count: {args.test_count}")
    print("=" * 60)
    print()

    # 运行评测
    run_psychobench(args, qwen25_14b_vllm_generator)

    print()
    print("=" * 60)
    print("PsychoBench评测完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
