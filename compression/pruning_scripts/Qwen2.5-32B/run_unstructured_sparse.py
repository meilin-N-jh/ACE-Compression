#!/usr/bin/env python3
"""
无结构化稀疏剪枝脚本 - Qwen2.5-32B
使用 llmcompressor 进行 50% 无结构化稀疏剪枝，使用 Alpaca 数据集校准

使用方法:
    conda activate pruning
    cd {ARTIFACT_ROOT}/pruning_scripts/Qwen2.5-32B
    python run_unstructured_sparse.py
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
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用第6张卡
import sys

# ==================== 配置 ====================
SOURCE_MODEL = f"{ARTIFACT_ROOT}/models/Qwen2.5-32B/Qwen2.5-32B-Instruct"
OUTPUT_MODEL = f"{ARTIFACT_ROOT}/models/Qwen2.5-32B/Qwen2.5-32B-Unstructured-Sparse-50"

NUM_SAMPLES = 128  # 标准配置：128 条样本
SEQ_LEN = 2048     # 标准配置：2048 tokens
SPARSITY = 0.5     # 50%
# ============================================

print("=" * 60)
print("无结构化稀疏剪枝 - Qwen2.5-32B (50%)")
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

# 兼容导入
try:
    from llmcompressor.modifiers.pruning import SparseGPTModifier
except ImportError:
    try:
        from llmcompressor.modifiers.quantization.gptq.base import SparseGPTModifier
    except ImportError:
        from llmcompressor import modifiers
        SparseGPTModifier = getattr(getattr(modifiers, 'pruning', None), 'SparseGPTModifier', None)
        if SparseGPTModifier is None:
            raise ImportError("无法导入 SparseGPTModifier")

print("\n加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    SOURCE_MODEL, torch_dtype="auto", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, trust_remote_code=True)


def get_calibration_data(tokenizer, num_samples=128, seq_len=2048):
    """使用 Alpaca 指令集进行校准，贴合 Instruct 模型的激活模式"""
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    samples = []

    for item in dataset:
        # 构建具有完整逻辑流的 Prompt
        text = f"Instruction: {item.get('instruction', '')}\n"
        if item.get('input', ''):
            text += f"Input: {item.get('input', '')}\n"
        text += f"Output: {item.get('output', '')}\n"

        enc = tokenizer(text, truncation=True, max_length=seq_len)

        # 过滤掉太短的样本
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


# 定义 recipe - 使用无结构化稀疏
recipe = SparseGPTModifier(
    sparsity=SPARSITY,
    mask_structure="0:0",  # 无结构化稀疏
    sequential_update=True,
    targets=[
        "re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj",
        "re:.*gate_proj", "re:.*up_proj", "re:.*down_proj"
    ]
)

print(f"\n开始剪枝 (sparsity={SPARSITY}, mask_structure=0:0 无结构化)...")
print(f"校准样本数: {NUM_SAMPLES}, 序列长度: {SEQ_LEN}")

# 释放内存
import torch
torch.cuda.empty_cache()
calibration_data = get_calibration_data(tokenizer, num_samples=NUM_SAMPLES, seq_len=SEQ_LEN)
del calibration_data
torch.cuda.empty_cache()

oneshot(
    model=model,
    dataset=calibration_data,
    recipe=recipe,
    output_dir=OUTPUT_MODEL,
    num_calibration_samples=len(calibration_data),
    max_seq_length=SEQ_LEN,
    batch_size=1,
    save_compressed=False  # 禁用压缩保存
)

tokenizer.save_pretrained(OUTPUT_MODEL)
print(f"\n✓ 剪枝完成: {OUTPUT_MODEL}")
print(f"下一步: bash {ARTIFACT_ROOT}/models/Qwen2.5-32B/start_vllm_unstructured.sh")
