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

# Run ToM-Bench for DeepSeek-V2-Lite-16B-GPTQ-INT4

python3 ${ARTIFACT_ROOT}/paper_benchmarks/07_ToM-Bench/benchmark/deepseek-v2-lite-16b/run_tombench_deepseek_vllm.py \
    --model-display-name "DeepSeek-V2-Lite-16B-GPTQ-INT4" \
    --base-url "http://127.0.0.1:8303" \
    --model-name "deepseek-v2-lite-gptq-int4" \
    --language zh \
    --try-times 5 \
    --seed 42
