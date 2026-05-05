#!/usr/bin/env python3
"""
2:4 半结构化稀疏剪枝脚本 - Qwen2.5-7B
使用 llmcompressor 进行 2:4 稀疏化，使用 Alpaca 数据集校准

使用方法:
    conda activate pruning
    cd {ARTIFACT_ROOT}/pruning_scripts/Qwen2.5-7B
    python run_2_4_sparse.py
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第0张卡
import sys
import torch
import gc

# ==================== 配置 ====================
SOURCE_MODEL = f"{ARTIFACT_ROOT}/models/Qwen2.5-7B/Qwen2.5-7B-Instruct"
OUTPUT_MODEL = f"{ARTIFACT_ROOT}/models/Qwen2.5-7B/Qwen2.5-7B-Instruct-2to4-Sparse"

NUM_SAMPLES = 128
SEQ_LEN = 2048
SPARSITY = 0.5
# ============================================

print("=" * 60)
print("2:4 半结构化稀疏剪枝 - Qwen2.5-7B")
print("=" * 60)
print(f"源模型: {SOURCE_MODEL}")
print(f"输出模型: {OUTPUT_MODEL}")
print("校准数据集: yahma/alpaca-cleaned")
print("=" * 60)

if not os.path.exists(SOURCE_MODEL):
    print(f"错误: 源模型不存在: {SOURCE_MODEL}")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier

print("\n加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    SOURCE_MODEL,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, trust_remote_code=True)


def get_calibration_data(tokenizer, num_samples=128, seq_len=2048):
    """使用 Alpaca 指令集进行校准"""
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    samples = []

    for item in dataset:
        text = f"Instruction: {item.get('instruction', '')}\n"
        if item.get('input', ''):
            text += f"Input: {item.get('input', '')}\n"
        text += f"Output: {item.get('output', '')}\n"

        enc = tokenizer(text, truncation=True, max_length=seq_len)
        if len(enc["input_ids"]) < 50:
            continue

        samples.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        })

        if len(samples) >= num_samples:
            break

    print(f"✅ 成功加载 {len(samples)} 条 Alpaca 数据用于校准。")
    return Dataset.from_list(samples)


# 准备校准数据
calibration_data = get_calibration_data(tokenizer, num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)

# 定义 recipe
recipe = SparseGPTModifier(
    sparsity=SPARSITY,
    mask_structure="2:4",
    sequential_update=True,
    targets=[
        "re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj",
        "re:.*gate_proj", "re:.*up_proj", "re:.*down_proj"
    ]
)

print(f"\n开始剪枝 (sparsity={SPARSITY}, mask_structure=2:4)...")
print(f"校准样本数: {NUM_SAMPLES}, 序列长度: {SEQ_LEN}")

oneshot(
    model=model,
    dataset=calibration_data,
    recipe=recipe,
    output_dir=OUTPUT_MODEL,
    num_calibration_samples=len(calibration_data),
    max_seq_length=SEQ_LEN,
    batch_size=1,
    save_compressed=False
)

# 删除 quantization_config，避免 vLLM 加载时报错
if hasattr(model.config, "quantization_config"):
    del model.config.quantization_config

tokenizer.save_pretrained(OUTPUT_MODEL)
print(f"\n✓ 剪枝完成: {OUTPUT_MODEL}")
print(f"下一步: bash {ARTIFACT_ROOT}/models/Qwen2.5-7B/start_vllm_sparse.sh")
