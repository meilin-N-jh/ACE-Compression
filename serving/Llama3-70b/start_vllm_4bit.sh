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

# 启动vLLM OpenAI API服务器 - bitsandbytes 4bit模型

export CUDA_VISIBLE_DEVICES=2
MODEL_PATH="${ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct"
HOST="0.0.0.0"
PORT="8002"
SERVED_MODEL_NAME="llama31-70b-bnb4bit"

echo "=========================================="
echo "启动vLLM OpenAI API服务器"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "量化: bitsandbytes 4bit "
echo "地址: http://$HOST:$PORT"
echo "=========================================="
echo ""


conda run -n llama python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --quantization bitsandbytes \
    --host $HOST \
    --port $PORT \
    --max-model-len 4096 \
    --dtype float16 \
    --trust-remote-code \
    --disable-log-stats
