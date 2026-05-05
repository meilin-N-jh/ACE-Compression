#!/usr/bin/env python3
"""
Llama-70B FP16 PsychoBench 评测
使用 GPU 0,1
"""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())


import sys
import os

# 添加父目录到 Python 路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from utils import *
from llama70b_generator import llama70b_generator

if __name__ == '__main__':
    import argparse
    
    class Args:
        model_path = f"{ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct"
        questionnaire = "BFI"
        quantization = "none"
        device = "auto"
        torch_dtype = "float16"
        shuffle_count = 1
        test_count = 10
        name_exp = "Llama-70B-FP16"
        significance_level = 0.01
        mode = "auto"
        model = "llama-70b"
    
    args = Args()
    
    print("=" * 80)
    print("PsychoBench Evaluation for Llama-70B FP16")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Questionnaire: {args.questionnaire}")
    print(f"Quantization: {args.quantization}")
    print(f"Device: {args.device} (GPU 0,1)")
    print(f"Shuffle count: {args.shuffle_count}")
    print(f"Test count: {args.test_count}")
    print("=" * 80)
    print()
    
    run_psychobench(args, llama70b_generator)
