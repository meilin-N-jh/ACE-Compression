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

# 启动vLLM OpenAI API服务器 - Llama3.1-8b GPTQ-INT4

export CUDA_VISIBLE_DEVICES=3
MODEL_PATH="${ARTIFACT_ROOT}/models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
HOST="0.0.0.0"
PORT="8403"
SERVED_MODEL_NAME="meta-llama-3.1-8b-instruct-gptq-int4"

echo "=========================================="
echo "启动vLLM OpenAI API服务器 - Llama3.1-8b GPTQ-INT4"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "地址: http://$HOST:$PORT"
echo "=========================================="

conda run -n llama python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host $HOST \
    --port $PORT \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-stats
