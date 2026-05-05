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

# 启动vLLM OpenAI API服务器 - AWQ INT4模型

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES
MODEL_PATH="${MODEL_PATH:-${ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct-AWQ-INT4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama31-70b-awq-int4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

echo "=========================================="
echo "启动vLLM OpenAI API服务器"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "Served Model Name: $SERVED_MODEL_NAME"
echo "量化: AWQ INT4"
echo "GPU: $CUDA_VISIBLE_DEVICES (1 张卡)"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "地址: http://$HOST:$PORT"
echo "=========================================="
echo ""

conda run -n awq python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --quantization awq \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --host $HOST \
    --port $PORT \
    --max-model-len 4096 \
    --dtype float16 \
    --trust-remote-code \
    --disable-log-stats
