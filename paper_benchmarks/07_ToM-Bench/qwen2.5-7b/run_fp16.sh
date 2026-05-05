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

# Run ToM-Bench for Qwen2.5-7B-FP16

cd ${ARTIFACT_ROOT}/paper_benchmarks/07_ToM-Bench/benchmark/qwen2.5-7b
python3 run_tombench_qwen25_vllm.py \
    --model-display-name "Qwen2.5-7B-FP16" \
    --base-url "http://127.0.0.1:8400" \
    --model-name "qwen2.5-7b-fp16" \
    --language zh \
    --try-times 5 \
    --seed 42
