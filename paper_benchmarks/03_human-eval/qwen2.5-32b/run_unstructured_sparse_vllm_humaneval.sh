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
# Script: run_unstructured_sparse_vllm_humaneval.sh
# Description: Evaluate Qwen2.5-32B-Instruct (Unstructured Sparse) on HumanEval
# Env: qwen2.5 (conda)
# GPU: GPU 4
# Output: paper_benchmarks/03_human-eval/benchmark/qwen2.5-32b/results/
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Qwen2.5-32B/Qwen2.5-32B-Unstructured-Sparse-50"
OUT_DIR="$BASE/paper_benchmarks/03_human-eval/benchmark/qwen2.5-32b/results"
LOG_DIR="$BASE/paper_benchmarks/03_human-eval/benchmark/qwen2.5-32b/logs"

LIMIT="${LIMIT:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL="1"

mkdir -p "$OUT_DIR" "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/humaneval_unstructured_sparse_vllm.json"
LOG_FILE="$LOG_DIR/humaneval_unstructured_sparse_vllm_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting HumanEval - Qwen2.5-32B-Instruct (Unstructured Sparse)"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
  --tasks humaneval_instruct
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON"
  --log_samples
  --verbosity INFO
  --confirm_run_unsafe_code
)

[[ "$LIMIT" != "0" && "$LIMIT" != "" ]] && args+=(--limit "$LIMIT")

echo "Command: lm-eval" "${args[@]}"
time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args[@]}"

echo "Results saved to $OUT_JSON"
