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


# -----------------------------------------------------------------------------
# Script: run_unstructured_sparse_vllm_ceval.sh
# Description: Evaluate DeepSeek-V2-Lite-16B (Unstructured Sparse) on C-Eval
# Env: qwen2.5
# GPU: GPU 6
# Output: paper_benchmarks/05_C-eval/launchers/deepseek-v2-lite-16b/results/
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-Unstructured-Sparse-50"
OUT_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/deepseek-v2-lite-16b/results"
LOG_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/deepseek-v2-lite-16b/logs"

LIMIT="${LIMIT:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
LOG_FILE="$LOG_DIR/ceval_unstructured_sparse_vllm_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting C-Eval - DeepSeek-V2-Lite-16B (Unstructured Sparse)"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

# Valid split
echo "Running C-Eval VALID split..."
OUT_JSON_VALID="$OUT_DIR/ceval_valid_unstructured_sparse_vllm_${TS}.json"

args_valid=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
  --tasks ceval-valid
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON_VALID"
  --verbosity INFO
)

[[ "$LIMIT" != "0" && "$LIMIT" != "" ]] && args_valid+=(--limit "$LIMIT")

time python -m lm_eval "${args_valid[@]}"
echo "Valid Results saved to $OUT_JSON_VALID"

# Test split
echo "Running C-Eval TEST split..."
OUT_JSON_TEST="$OUT_DIR/ceval_test_unstructured_sparse_vllm_${TS}.json"

args_test=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
  --include_path "$BASE/paper_benchmarks/05_C-eval/launchers/tasks/ceval-test"
  --tasks ceval-test
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON_TEST"
  --verbosity INFO
)

[[ "$LIMIT" != "0" && "$LIMIT" != "" ]] && args_test+=(--limit "$LIMIT")

time python -m lm_eval "${args_test[@]}"
echo "Test Results saved to $OUT_JSON_TEST"

echo "All Evaluations Finished."
