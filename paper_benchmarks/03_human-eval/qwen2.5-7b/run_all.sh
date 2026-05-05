#!/usr/bin/env bash
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


# Run all HumanEval pass@10 & pass@100 for qwen2.5-7b
BASE="${ARTIFACT_ROOT}"
cd "$BASE/paper_benchmarks/03_human-eval/benchmark/qwen2.5-7b"

echo "Running qwen2.5-7b all variants..."

CUDA_VISIBLE_DEVICES=0 LIMIT=0 bash run_fp16_vllm_humaneval_pass10_100.sh
CUDA_VISIBLE_DEVICES=0 LIMIT=0 bash run_awq_vllm_humaneval_pass10_100.sh
CUDA_VISIBLE_DEVICES=0 LIMIT=0 bash run_gptq_int4_vllm_humaneval_pass10_100.sh
CUDA_VISIBLE_DEVICES=0 LIMIT=0 bash run_bnb_4bit_vllm_humaneval_pass10_100.sh
CUDA_VISIBLE_DEVICES=0 LIMIT=0 bash run_2_4_sparse_vllm_humaneval_pass10_100.sh
CUDA_VISIBLE_DEVICES=0 LIMIT=0 bash run_unstructured_sparse_vllm_humaneval_pass10_100.sh

echo "Done!"
