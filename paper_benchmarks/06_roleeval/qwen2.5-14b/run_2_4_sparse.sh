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

# 运行 2:4 Sparse 模型的 RoleEval 评测

MODEL_KEY="qwen25_14b_2_4_sparse_vllm"

cd ${ARTIFACT_ROOT}/paper_benchmarks/06_roleeval/benchmark/qwen2.5-14b

echo "=========================================="
echo "RoleEval - Qwen2.5-14B 2:4 Sparse"
echo "=========================================="

python run_roleeval_qwen25_14b.py \
    --model "$MODEL_KEY" \
    --lang both \
    --subset both \
    --split test

echo "结果保存在: $(pwd)/results/"
