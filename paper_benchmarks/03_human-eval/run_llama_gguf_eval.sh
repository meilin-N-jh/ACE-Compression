#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_artifact_root() {
  local dir="$SCRIPT_DIR"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/configs/models.yaml" ]]; then
      printf "%s\n" "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$(find_artifact_root || true)}"
if [[ -z "${ARTIFACT_ROOT}" ]]; then
  echo "Could not locate artifact root (configs/models.yaml)." >&2
  exit 1
fi


# HumanEval GGUF模型评测运行脚本
# 在llama-gguf环境中运行

# 激活llama-gguf环境
conda activate llama-gguf

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1

# 进入human-eval目录
cd ${ARTIFACT_ROOT}/paper_benchmarks/03_human-eval/benchmark

echo "========================================"
echo "HumanEval GGUF模型评测 (llama-gguf环境)"
echo "========================================"
echo "环境: llama-gguf"
echo "GPU: 1号"
echo "模型: Qwen-7B-Chat.gguf"
echo "========================================"

# 运行GGUF评测脚本
python run_gguf_evaluation.py \
    --model-path ${ARTIFACT_ROOT}/models/Qwen-7B-Chat-GGUF/Qwen-7B-Chat.Q4_K_M.gguf \
    --samples 1 \
    --temperature 0.1 \
    --max-tokens 256 \
    --n-gpu-layers 35 \
    --n-ctx 4096 \
    --output qwen_gguf_human_eval_samples_fixed.jsonl

# 检查评测是否成功
if [ $? -eq 0 ]; then
    echo "✓ GGUF评测完成!"
    echo "输出文件: qwen_gguf_human_eval_samples.jsonl"

    # 运行功能正确性评测
    echo ""
    echo "开始运行功能正确性评测..."
    evaluate_functional_correctness qwen_gguf_human_eval_samples.jsonl
else
    echo "✗ GGUF评测失败!"
    exit 1
fi

echo ""
echo "========================================"
echo "HumanEval GGUF评测完成"
echo "========================================"