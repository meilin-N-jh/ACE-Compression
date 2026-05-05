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

source /opt/anaconda3/etc/profile.d/conda.sh || true
conda activate awq

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model ${ARTIFACT_ROOT}/models/Qwen2.5-14B/Qwen2.5-14B-Instruct-AWQ \
    --served-model-name "Qwen2.5-14B-Instruct-AWQ" \
    --quantization awq \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --port 8102 \
    --dtype float16
