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
# Script: run_unstructured_sparse_vllm_gsm8k.sh
# Description: Evaluate DeepSeek-V2-Lite-16B (Unstructured Sparse) on GSM8K
# Env: qwen2.5
# GPU: GPU 6
# Output: paper_benchmarks/02_gsm8k/launchers/deepseek-v2-lite-16b/results/
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-Unstructured-Sparse-50"
OUT_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/deepseek-v2-lite-16b/results"
LOG_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/deepseek-v2-lite-16b/logs"

LIMIT="${LIMIT:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/gsm8k_cot_unstructured_sparse_vllm_${TS}.json"
LOG_FILE="$LOG_DIR/gsm8k_cot_unstructured_sparse_vllm_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting GSM8K - DeepSeek-V2-Lite-16B (Unstructured Sparse)"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,dtype=float16,trust_remote_code=True"
  --tasks gsm8k_cot
  --apply_chat_template
  --fewshot_as_multiturn
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON"
  --log_samples
  --verbosity INFO
)

[[ "$LIMIT" != "0" && "$LIMIT" != "" ]] && args+=(--limit "$LIMIT")

echo "Command: lm-eval" "${args[@]}"
time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args[@]}"

echo "Results saved to $OUT_JSON"
