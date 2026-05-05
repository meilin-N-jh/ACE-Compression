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
# Script: run_gptq_vllm_ceval.sh
# Description: Evaluate Meta-Llama-3.1-8B-Instruct (GPTQ-INT4) on C-Eval (Valid & Test)
# Env: llama (conda)
# GPU: GPU 0
# Output: paper_benchmarks/05_C-eval/launchers/llama3.1-8b/results/ and results_valid/
# Notes:
#   - Runs both ceval-valid and ceval-test in one script
#   - Default runs a smoke test with LIMIT=10.
#   - For full run: LIMIT=0 ./run_gptq_vllm_ceval.sh
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
OUT_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/llama3.1-8b/results"
OUT_DIR_VALID="$BASE/paper_benchmarks/05_C-eval/launchers/llama3.1-8b/results_valid"
LOG_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/llama3.1-8b/logs"

LIMIT="${LIMIT:-10}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR_VALID"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
LOG_FILE="$LOG_DIR/ceval_gptq_vllm_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting C-Eval Evaluation - Meta-Llama-3.1-8B-Instruct (GPTQ-INT4)"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LIMIT=$LIMIT"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

# 1. Run Validation Split (ceval-valid)
echo "--------------------------------------------------------"
echo "Running C-Eval VALID split..."
OUT_JSON_VALID="$OUT_DIR_VALID/ceval_valid_gptq_vllm_${TS}.json"

args_valid=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,gpu_memory_utilization=0.9,max_model_len=4096,quantization=gptq,trust_remote_code=True"
  --tasks ceval-valid
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON_VALID"
  --verbosity INFO
)

if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args_valid+=(--limit "$LIMIT")
fi

echo "Command (Valid): lm-eval" "${args_valid[@]}"
time conda run --no-capture-output -n llama python -m lm_eval "${args_valid[@]}"
echo "Valid Results saved to $OUT_JSON_VALID"

# 2. Run Test Split (ceval-test)
echo "--------------------------------------------------------"
echo "Running C-Eval TEST split..."
OUT_JSON_TEST="$OUT_DIR/ceval_test_gptq_vllm_${TS}.json"

args_test=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,gpu_memory_utilization=0.9,max_model_len=4096,quantization=gptq,trust_remote_code=True"
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
time conda run --no-capture-output -n llama python -m lm_eval "${args_test[@]}"
echo "Test Results saved to $OUT_JSON_TEST"

echo "========================================================"
echo "All Evaluations Finished."
echo "Log saved to $LOG_FILE"
echo "========================================================"
