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
# Script: run_unstructured_sparse_vllm_humaneval_pass10_100.sh
# Description: Evaluate Llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse-50 (unstructured_sparse) on HumanEval pass@10 and pass@100
# Env: qwen2.5 (conda)
# GPU: GPU 4 (tensor_parallel_size=1)
# Output: paper_benchmarks/03_human-eval/benchmark/llama3.1-8b/results/
# Notes:
#   - Runs pass@10 first, then pass@100 automatically
#   - Uses lm-eval vLLM backend with unstructured_sparse precision
#   - Model: Llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse-50
#   - Default runs a smoke test with LIMIT=10
#   - For full run: LIMIT=0 ./run_unstructured_sparse_vllm_humaneval_pass10_100.sh
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse-50"
OUT_DIR="$BASE/paper_benchmarks/03_human-eval/benchmark/llama3.1-8b/results"
LOG_DIR="$BASE/paper_benchmarks/03_human-eval/benchmark/llama3.1-8b/logs"

# LIMIT=10 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-10}"

# Use GPU 4 (tensor_parallel_size=1)
export CUDA_VISIBLE_DEVICES="4"

# Model weights are local; keep transformers offline.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
# Allow code evaluation for HumanEval
export HF_ALLOW_CODE_EVAL="1"

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
BATCH_SIZE="${BATCH_SIZE:-auto}"

echo "========================================================"
echo "Starting HumanEval pass@10 & pass@100 - Llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse-50 (unstructured_sparse)"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Tensor Parallel Size: 1"
echo "LIMIT=$LIMIT"
echo "========================================================"

# ========================================================
# Step 1: Run pass@10
# ========================================================
echo ""
echo "========================================================"
echo "Step 1/2: Running pass@10 evaluation..."
echo "========================================================"

PASS10_OUT_JSON="$OUT_DIR/humaneval_unstructured_sparse_vllm_pass10_${TS}.json"
PASS10_LOG_FILE="$LOG_DIR/humaneval_unstructured_sparse_vllm_pass10_${TS}.log"

exec > >(tee -a "$PASS10_LOG_FILE") 2>&1

args_pass10=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=4096,dtype=float16,trust_remote_code=True,max_num_seqs=256,max_num_batched_tokens=8192"
  --tasks humaneval_instruct_pass10
  --batch_size "$BATCH_SIZE"
  --output_path "$PASS10_OUT_JSON"
  --log_samples
  --verbosity INFO
  --confirm_run_unsafe_code
)

if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args_pass10+=(--limit "$LIMIT")
fi

echo "Command: lm-eval" "${args_pass10[@]}"
time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args_pass10[@]}"

echo "pass@10 completed! Results saved to: $PASS10_OUT_JSON"
echo "pass@10 log saved to: $PASS10_LOG_FILE"

# ========================================================
# Step 2: Run pass@100
# ========================================================
echo ""
echo "========================================================"
echo "Step 2/2: Running pass@100 evaluation..."
echo "========================================================"

PASS100_OUT_JSON="$OUT_DIR/humaneval_unstructured_sparse_vllm_pass100_${TS}.json"
PASS100_LOG_FILE="$LOG_DIR/humaneval_unstructured_sparse_vllm_pass100_${TS}.log"

# Redirect output to pass@100 log
exec > >(tee -a "$PASS100_LOG_FILE") 2>&1

args_pass100=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=4096,dtype=float16,trust_remote_code=True,max_num_seqs=256,max_num_batched_tokens=8192"
  --tasks humaneval_instruct_pass100
  --batch_size "$BATCH_SIZE"
  --output_path "$PASS100_OUT_JSON"
  --log_samples
  --verbosity INFO
  --confirm_run_unsafe_code
)

if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args_pass100+=(--limit "$LIMIT")
fi

echo "Command: lm-eval" "${args_pass100[@]}"
time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args_pass100[@]}"

echo "pass@100 completed! Results saved to: $PASS100_OUT_JSON"
echo "pass@100 log saved to: $PASS100_LOG_FILE"

# ========================================================
# Summary
# ========================================================
echo ""
echo "========================================================"
echo "Evaluation Complete!"
echo "========================================================"
echo "pass@10 Results: $PASS10_OUT_JSON"
echo "pass@10 Log: $PASS10_LOG_FILE"
echo "pass@100 Results: $PASS100_OUT_JSON"
echo "pass@100 Log: $PASS100_LOG_FILE"
echo "========================================================"
