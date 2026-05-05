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

# 运行单个模型的双语RoleEval评测
# 用法: ./run_single_model.sh <model_key>
#
# model_key 选项:
#   - qwen25_14b_fp16_vllm
#   - qwen25_14b_bnb_4bit_vllm
#   - qwen25_14b_awq_vllm
#   - qwen25_14b_gptq_int4_vllm
#   - qwen25_14b_trim_vllm

MODEL_KEY=${1:-"qwen25_14b_fp16_vllm"}

cd ${ARTIFACT_ROOT}/paper_benchmarks/06_roleeval/benchmark/qwen2.5-14b

echo "=========================================="
echo "RoleEval 双语评测 - 单个模型"
echo "=========================================="
echo "模型: $MODEL_KEY"
echo "语言: zh + en"
echo "子集: chinese + global"
echo "Split: test"
echo "=========================================="
echo ""

# 运行评测（需要先在终端激活conda环境: conda activate qwen2.5）
python run_roleeval_qwen25_14b.py \
    --model "$MODEL_KEY" \
    --lang both \
    --subset both \
    --split test

echo ""
echo "=========================================="
echo "评测完成！"
echo "=========================================="
echo "结果保存在: $(pwd)/results/"
echo "=========================================="
