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

# RoleEval - DeepSeek-V2-Lite 2:4 Sparse

MODEL_KEY="deepseek_v2_lite_2_4_sparse_vllm"

cd ${ARTIFACT_ROOT}/paper_benchmarks/06_roleeval/benchmark/deepseek-v2-lite-16b
python run_roleeval_deepseek.py --model "$MODEL_KEY" --lang both --subset both --split test
