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
# Script: run_fp16_vllm_ceval.sh
# Description: Evaluate Qwen2.5-14B-Instruct (FP16) on C-Eval (Valid & Test)
# Env: benchmark (custom)
# GPU: Single GPU (TP=1)
# Output: paper_benchmarks/05_C-eval/launchers/qwen2.5-14b/results/
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Qwen2.5-14B/Qwen2.5-14B-Instruct-FP16"
OUT_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/qwen2.5-14b/results"
LOG_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/qwen2.5-14b/logs"

# LIMIT=10 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-10}"

# Use 1 GPU for 14B model
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
# We will use distinct log files for clarity or append to one. Let's append.
LOG_FILE="$LOG_DIR/ceval_fp16_vllm_${TS}.log"

# Send all output to the log and console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting C-Eval Evaluation - Qwen2.5-14B-Instruct (FP16)"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LIMIT=$LIMIT"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

# 1. Run Validation Split (ceval-valid)
echo "--------------------------------------------------------"
echo "Running C-Eval VALID split..."
OUT_JSON_VALID="$OUT_DIR/ceval_valid_fp16_vllm_${TS}.json"

args_valid=(
  --model vllm
  # TP=1 for 14B model on standard GPU
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
  --tasks ceval-valid
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON_VALID"
  --verbosity INFO
)

if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args_valid+=(--limit "$LIMIT")
fi

echo "Command (Valid): lm-eval" "${args_valid[@]}"
time python -m lm_eval "${args_valid[@]}"
echo "Valid Results saved to $OUT_JSON_VALID"

# 2. Run Test Split (ceval-test)
echo "--------------------------------------------------------"
echo "Running C-Eval TEST split..."
OUT_JSON_TEST="$OUT_DIR/ceval_test_fp16_vllm_${TS}.json"

args_test=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
  # Include path for custom ceval-test tasks
  --include_path "$BASE/paper_benchmarks/05_C-eval/launchers/tasks/ceval-test"
  --tasks ceval-test
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON_TEST"
  --verbosity INFO
)

if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args_test+=(--limit "$LIMIT")
fi

echo "Command (Test): lm-eval" "${args_test[@]}"
time python -m lm_eval "${args_test[@]}"
echo "Test Results saved to $OUT_JSON_TEST"

echo "========================================================"
echo "All Evaluations Finished."
echo "Log saved to $LOG_FILE"
echo "========================================================"
