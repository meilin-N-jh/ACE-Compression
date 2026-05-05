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
# 用法: bash run_2_4_sparse.sh

MODEL_KEY="qwen25_7b_2_4_sparse_vllm"

cd ${ARTIFACT_ROOT}/paper_benchmarks/06_roleeval/benchmark/qwen2.5-7b

echo "=========================================="
echo "RoleEval 双语评测 - 2:4 Sparse"
echo "=========================================="
echo "模型: $MODEL_KEY"
echo "语言: zh + en"
echo "子集: chinese + global"
echo "Split: test"
echo "=========================================="
echo ""

python run_roleeval_qwen25_7b.py \
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
