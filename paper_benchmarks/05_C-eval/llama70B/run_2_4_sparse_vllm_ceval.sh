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
# Script: run_2_4_sparse_vllm_ceval.sh
# Description: Evaluate Llama-3.1-70B-Instruct (2:4 Sparse) on C-Eval (Valid & Test)
# Env: llama (conda)
# GPU: GPUs 0,1
# Output: paper_benchmarks/05_C-eval/launchers/llama70B/results/ and results_valid/
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Llama3-70b/Llama-3.1-70B-Instruct-2to4-Sparse"
OUT_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/llama70B/results"
OUT_DIR_VALID="$BASE/paper_benchmarks/05_C-eval/launchers/llama70B/results_valid"
LOG_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/llama70B/logs"

# LIMIT=10 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-10}"
VALID_ONLY="${VALID_ONLY:-0}"

# Use 2 GPUs (GPU 0,1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR_VALID"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
LOG_FILE="$LOG_DIR/ceval_2_4_sparse_vllm_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting C-Eval - Llama-3.1-70B-Instruct (2:4 Sparse) with vLLM"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LIMIT=$LIMIT"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

# 1. Run Validation Split (ceval-valid)
echo "--------------------------------------------------------"
echo "Running C-Eval VALID split..."
OUT_JSON_VALID="$OUT_DIR_VALID/ceval_valid_2_4_sparse_vllm_${TS}.json"

args_valid=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=2,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
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

if [[ "$VALID_ONLY" == "1" ]]; then
  echo "========================================================"
  echo "VALID_ONLY=1, skipping TEST split"
  echo "========================================================"
else
# 2. Run Test Split (ceval-test)
echo "--------------------------------------------------------"
echo "Running C-Eval TEST split..."
OUT_JSON_TEST="$OUT_DIR/ceval_test_2_4_sparse_vllm_${TS}.json"

args_test=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=2,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
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

fi

echo "========================================================"
echo "All Evaluations Finished."
echo "Log saved to $LOG_FILE"
echo "========================================================"
