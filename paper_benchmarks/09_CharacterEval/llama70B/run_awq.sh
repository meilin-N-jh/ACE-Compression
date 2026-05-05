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

# Run CharacterEval for Llama-70B-AWQ

cd ${ARTIFACT_ROOT}/paper_benchmarks/09_CharacterEval/benchmark

python llama70B/get_response_vllm.py \
    --model-display-name "Llama-70B-AWQ" \
    --base-url "http://127.0.0.1:8000" \
    --model-name "llama31-70b-awq-int4"
