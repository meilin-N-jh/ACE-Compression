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

# 启动vLLM OpenAI API服务器 - Llama3-70b 无结构化稀疏模型

export CUDA_VISIBLE_DEVICES=6,7
MODEL_PATH="${ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct-Unstructured-Sparse-50"
HOST="0.0.0.0"
PORT="8005"
SERVED_MODEL_NAME="llama3-70b-unstructured-sparse"
TENSOR_PARALLEL_SIZE=2

echo "=========================================="
echo "启动vLLM - Llama3-70b 无结构化稀疏 (50%)"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "端口: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL_SIZE"
echo "=========================================="

conda run -n qwen2.5 python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --host $HOST \
    --port $PORT \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 16384 \
    --disable-log-stats
