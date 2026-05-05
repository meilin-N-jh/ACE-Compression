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

# 启动vLLM - DeepSeek-V2-Lite Unstructured Sparse

export CUDA_VISIBLE_DEVICES=1
MODEL_PATH="${ARTIFACT_ROOT}/models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-Unstructured-Sparse-50"
PORT="8305"
SERVED_MODEL_NAME="deepseek-v2-lite-unstructured-sparse"

conda run -n qwen2.5 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len 8192 \
    --dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-stats
