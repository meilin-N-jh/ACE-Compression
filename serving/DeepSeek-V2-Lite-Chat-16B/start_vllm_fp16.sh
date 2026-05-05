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

# 启动vLLM OpenAI API服务器 - DeepSeek-V2-Lite-16B FP16模型

export CUDA_VISIBLE_DEVICES=2
MODEL_PATH="${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat"
HOST="0.0.0.0"
PORT="8300"
SERVED_MODEL_NAME="deepseek-v2-lite-fp16"
MAX_MODEL_LEN="8192"

echo "=========================================="
echo "启动vLLM OpenAI API服务器"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "地址: http://$HOST:$PORT"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "=========================================="
echo ""

conda run -n qwen2.5 python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host $HOST \
    --port $PORT \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 32768 \
    --disable-log-stats
