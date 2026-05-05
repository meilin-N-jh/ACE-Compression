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
# Script: run_gptq_int8_vllm_gsm8k.sh
# Description: Evaluate Llama-3.1-70B-Instruct (GPTQ INT8) on GSM8K using lm-eval vllm backend
# Env: llama (conda)
# GPU: GPUs 0,1 (tensor parallel)
# Output: paper_benchmarks/02_gsm8k/launchers/llama70B/results/
# Notes:
#   - Uses lm-eval vLLM backend with GPTQ INT8 quantization
#   - Model: Meta-Llama-3.1-70B-Instruct-GPTQ-INT8
#   - Default runs a smoke test with LIMIT=1.
#   - For full run: LIMIT=0 ./run_gptq_int8_vllm_gsm8k.sh
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Llama3-70b/Meta-Llama-3.1-70B-Instruct-GPTQ-INT8"
OUT_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/llama70B/results"
LOG_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/llama70B/logs"

# LIMIT=1 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-1}"

# Use 2 GPUs with tensor parallelism
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Model weights are local; keep transformers offline.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/gsm8k_cot_gptq_int8_vllm_${TS}.json"
LOG_FILE="$LOG_DIR/gsm8k_cot_gptq_int8_vllm_${TS}.log"

# Send *all* output to the log and console.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting GSM8K (gsm8k_cot) - Llama-3.1-70B-Instruct GPTQ INT8 with vLLM"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LIMIT=$LIMIT"
echo "Output: $OUT_JSON"
echo "Log: $LOG_FILE"
echo "========================================================"

# vLLM backend with GPTQ INT8 quantization
BATCH_SIZE="${BATCH_SIZE:-auto}"

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=2,gpu_memory_utilization=0.9,max_model_len=4096,quantization=gptq,dtype=auto,trust_remote_code=True"
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

time conda run --no-capture-output -n llama python -m lm_eval "${args[@]}"

echo "========================================================"
echo "Evaluation Finished."
echo "Results saved to $OUT_JSON"
echo "Log saved to $LOG_FILE"
echo "========================================================"
