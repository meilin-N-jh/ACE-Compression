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

# 启动vLLM OpenAI API服务器 - Qwen2.5-14B 2:4 Sparse模型

export CUDA_VISIBLE_DEVICES=7
MODEL_PATH="${ARTIFACT_ROOT}/models/Qwen2.5-14B/Qwen2.5-14B-2to4-Sparse"
HOST="0.0.0.0"
PORT="8104"
SERVED_MODEL_NAME="qwen2.5-14b-2to4-sparse"

echo "=========================================="
echo "启动vLLM - Qwen2.5-14B 2:4 Sparse"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "端口: $PORT"
echo "=========================================="

conda run -n qwen2.5 python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host $HOST \
    --port $PORT \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.98 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 32768 \
    --disable-log-stats
