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

# Run CharacterRM scoring for Qwen2.5-7B

cd ${ARTIFACT_ROOT}/paper_benchmarks/09_CharacterEval/benchmark/qwen2.5-7b

# Usage: bash run_char_rm.sh <model_name> [gpu]
# Example: bash run_char_rm.sh Qwen2.5-7B-FP16 0

MODEL_NAME=${1:-"Qwen2.5-7B-FP16"}
GPU=${2:-0}

python run_char_rm_vllm.py --model-name "$MODEL_NAME" --gpu $GPU
