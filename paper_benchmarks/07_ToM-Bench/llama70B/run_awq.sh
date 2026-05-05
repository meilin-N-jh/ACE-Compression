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

# Run ToM-Bench for Llama-70B-AWQ

cd ${ARTIFACT_ROOT}/paper_benchmarks/07_ToM-Bench/benchmark

python llama70B/run_tombench_llama70b_vllm.py \
    --model-display-name "Llama-70B-AWQ" \
    --base-url "http://127.0.0.1:8000" \
    --model-name "llama31-70b-awq-int4" \
    --language zh \
    --try-times 5 \
    --seed 42
