#!/usr/bin/env python3
"""
2:4 半结构化稀疏剪枝脚本 - Llama3-70b
使用 llmcompressor 进行 2:4 稀疏化

使用方法:
    conda activate pruning
    cd {ARTIFACT_ROOT}/pruning_scripts/Llama3-70b
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
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # 使用第6、7张卡
import sys
import torch
import gc

# ==================== 配置 ====================
SOURCE_MODEL = f"{ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct"
OUTPUT_MODEL = f"{ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct-2to4-Sparse"

NUM_SAMPLES = 128
SEQ_LEN = 2048
SPARSITY = 0.5
# ============================================

print("=" * 60)
print("2:4 半结构化稀疏剪枝 - Llama3-70b")
print("=" * 60)
print(f"源模型: {SOURCE_MODEL}")
print(f"输出模型: {OUTPUT_MODEL}")
print("=" * 60)

if not os.path.exists(SOURCE_MODEL):
    print(f"错误: 源模型不存在: {SOURCE_MODEL}")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import SparseGPTModifier

# 强制不平衡分配
max_memory_mapping = {
    0: "50GiB",   # 卡 0 只装 50G 权重，留 45G 空闲保命
    1: "93GiB",   # 卡 1 尽量塞满
}

print("INFO: 正在加载 70B 模型...")
model = AutoModelForCausalLM.from_pretrained(
    SOURCE_MODEL,
    torch_dtype="auto",
    device_map="auto",
    max_memory=max_memory_mapping
)
tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL)


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

print("INFO: 配置剪枝方案...")
recipe = SparseGPTModifier(
    sparsity=SPARSITY,
    mask_structure="2:4",
    sequential_update=True,
    targets=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj",
             "re:.*gate_proj", "re:.*up_proj", "re:.*down_proj"]
)

print(f"\n开始剪枝 (sparsity={SPARSITY}, mask_structure=2:4)...")
print(f"校准样本数: {NUM_SAMPLES}, 序列长度: {SEQ_LEN}")

print("INFO: 开始执行 oneshot 剪枝计算...")
oneshot(
    model=model,
    dataset=calibration_data,
    recipe=recipe,
    num_calibration_samples=len(calibration_data),
)

# ===============================================
# 【绝杀部分】：手动剥离并终结 llmcompressor 的缓存劫持
# ===============================================
print("INFO: 剪枝结束，启动终极防 OOM 剥离程序...")

if hasattr(model.config, "quantization_config"):
    del model.config.quantization_config

# 1. 强行删除 accelerate 挂载的设备地图
if hasattr(model, "hf_device_map"):
    delattr(model, "hf_device_map")

# 2. 遍历所有模块，解除 accelerate 和 compressed_tensors 的双重锁定
for name, module in model.named_modules():
    # 删除 accelerate 的 hook
    if hasattr(module, "_hf_hook"):
        delattr(module, "_hf_hook")

    # 【最关键的一步：篡改缓存转移目标】
    if hasattr(module, "_parameters") and hasattr(module._parameters, "onload_device"):
        module._parameters.onload_device = torch.device("cpu")

    if hasattr(module, "_buffers") and hasattr(module._buffers, "onload_device"):
        module._buffers.onload_device = torch.device("cpu")

# 3. 将所有参数安全、平滑地推入 CPU
print("INFO: 正在将模型参数搬运至 CPU 内存，这可能需要几分钟...")
model.cpu()

# 4. 彻底清空 GPU 显存
torch.cuda.empty_cache()
gc.collect()

# 5. 现在模型是一个纯净且不再反弹的 CPU 模型了
print("INFO: 开始在 CPU 内存中进行分片保存...")
model.save_pretrained(
    OUTPUT_MODEL,
    save_compressed=False,
    config=model.config,
    safe_serialization=True,
    max_shard_size="10GB"
)
tokenizer.save_pretrained(OUTPUT_MODEL)
print("INFO: 完美完成！")

print(f"\n✓ 剪枝完成: {OUTPUT_MODEL}")
