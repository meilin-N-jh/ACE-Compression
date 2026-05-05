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

# Run CharacterEval for Qwen2.5-14B-AWQ (vLLM OpenAI API)
# 需要先手动启动 vLLM 服务器

cd ${ARTIFACT_ROOT}/paper_benchmarks/09_CharacterEval/benchmark

python qwen2.5-14b/get_response_vllm.py \
    --model-display-name "Qwen2.5-14B-AWQ" \
    --base-url "http://127.0.0.1:8102" \
    --model-name "qwen2.5-14b-awq"
