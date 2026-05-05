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
# Script: run_trim_vllm_ceval.sh
# Description: Evaluate Qwen2.5-14B-Instruct-Trim on C-Eval (Valid & Test)
# Env: benchmark (custom)
# GPU: Single GPU (TP=1)
# Output: paper_benchmarks/05_C-eval/launchers/qwen2.5-14b/results/
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Qwen2.5-14B/qwen2.5-14b-instruct-trim"
OUT_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/qwen2.5-14b/results"
LOG_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/qwen2.5-14b/logs"

LIMIT="${LIMIT:-10}"
# GPU 4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
LOG_FILE="$LOG_DIR/ceval_trim_vllm_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting C-Eval Evaluation - Qwen2.5-14B-Instruct-Trim"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LIMIT=$LIMIT"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

# 1. Run Validation Split
echo "--------------------------------------------------------"
echo "Running C-Eval VALID split..."
OUT_JSON_VALID="$OUT_DIR/ceval_valid_trim_vllm_${TS}.json"

args_valid=(
  --model vllm
  # Assuming Trim is standard FP16/BF16
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

# 2. Run Test Split
echo "--------------------------------------------------------"
echo "Running C-Eval TEST split..."
OUT_JSON_TEST="$OUT_DIR/ceval_test_trim_vllm_${TS}.json"

args_test=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
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

echo "========================================================"
echo "All Evaluations Finished."
echo "Log saved to $LOG_FILE"
echo "========================================================"
