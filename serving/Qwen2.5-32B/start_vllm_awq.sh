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

# 启动vLLM OpenAI API服务器 - Qwen2.5-32B AWQ模型

export CUDA_VISIBLE_DEVICES=2
MODEL_PATH="${ARTIFACT_ROOT}/models/Qwen2.5-32B/Qwen2.5-32B-Instruct-AWQ"
HOST="0.0.0.0"
PORT="8202"
SERVED_MODEL_NAME="qwen2.5-32b-awq"

echo "=========================================="
echo "启动vLLM OpenAI API服务器"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "量化: AWQ"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "地址: http://$HOST:$PORT"
echo "=========================================="
echo ""

conda run -n qwen2.5 python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --quantization awq \
    --host $HOST \
    --port $PORT \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.50 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 32768 \
    --disable-log-stats
