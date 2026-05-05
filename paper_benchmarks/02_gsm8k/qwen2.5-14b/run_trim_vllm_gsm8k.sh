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
# Script: run_trim_vllm_gsm8k.sh
# Description: Evaluate Qwen2.5-14B-Instruct (Pruned) on GSM8K using lm-eval vllm backend
# Env: qwen2.5 (conda)
# GPU: GPU 4
# Output: paper_benchmarks/02_gsm8k/launchers/qwen2.5-14b/results/
# Notes:
#   - Uses lm-eval vllm backend for Pruned model
#   - Default runs a smoke test with LIMIT=1.
#   - For full run: LIMIT=0 ./run_trim_vllm_gsm8k.sh
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Qwen2.5-14B/qwen2.5-14b-instruct-trim"
OUT_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/qwen2.5-14b/results"
LOG_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/qwen2.5-14b/logs"

# LIMIT=1 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-1}"

# Use 1 GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# Model weights are local; keep transformers offline.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/gsm8k_cot_trim_vllm_${TS}.json"
LOG_FILE="$LOG_DIR/gsm8k_cot_trim_vllm_${TS}.log"

# Send *all* output to the log and console.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting GSM8K (gsm8k_cot) - Qwen2.5-14B-Instruct Pruned with vLLM"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LIMIT=$LIMIT"
echo "Output: $OUT_JSON"
echo "Log: $LOG_FILE"
echo "========================================================"

# vLLM backend arguments for Pruned model
BATCH_SIZE="${BATCH_SIZE:-auto}"

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.98,max_model_len=4096,dtype=auto,trust_remote_code=True,max_num_seqs=1024,max_num_batched_tokens=32768"
  --tasks gsm8k_cot
  --apply_chat_template
  --fewshot_as_multiturn
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON"
  --log_samples
  --verbosity INFO
)

if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args+=(--limit "$LIMIT")
fi

echo "Starting evaluation..."
echo "Command: lm-eval" "${args[@]}"
echo "========================================================"

time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args[@]}"

echo "========================================================"
echo "Evaluation Finished."
echo "Results saved to $OUT_JSON"
echo "Log saved to $LOG_FILE"
echo "========================================================"
