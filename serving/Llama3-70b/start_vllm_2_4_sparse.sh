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

# 启动vLLM - Llama70B 2:4 Sparse (TP=2)

export CUDA_VISIBLE_DEVICES=0,1
MODEL_PATH="${ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct-2to4-Sparse"
PORT="8010"
SERVED_MODEL_NAME="llama31-70b-2to4-sparse"

conda run -n llama python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --dtype float16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-stats
