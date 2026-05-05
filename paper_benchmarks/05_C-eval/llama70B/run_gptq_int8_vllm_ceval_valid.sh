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
# Script: run_gptq_int8_vllm_ceval.sh
# Description: Evaluate Llama-3.1-70B-Instruct (GPTQ INT8) on C-Eval using lm-eval vllm backend
# Env: llama (conda)
# GPU: GPU 3 (single GPU)
# Output: paper_benchmarks/05_C-eval/launchers/llama70B/results/
# Notes:
#   - Uses lm-eval vLLM backend with GPTQ INT8 quantization
#   - Model: Meta-Llama-3.1-70B-Instruct-GPTQ-INT8
#   - Task: ceval-valid (52 Chinese subjects)
#   - Default runs a smoke test with LIMIT=10.
#   - For full run: LIMIT=0 ./run_gptq_int8_vllm_ceval.sh
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Llama3-70b/Meta-Llama-3.1-70B-Instruct-GPTQ-INT8"
OUT_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/llama70B/results"
LOG_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/llama70B/logs"

# LIMIT=10 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-10}"

# Use single GPU (GPU 3)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Model weights are local; keep transformers offline.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/ceval_gptq_int8_vllm_${TS}.json"
LOG_FILE="$LOG_DIR/ceval_gptq_int8_vllm_${TS}.log"

# Send *all* output to the log and console.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting C-Eval (VALID) - Llama-3.1-70B-Instruct GPTQ INT8 with vLLM"
echo "Date: $(date)"
echo "Model: $MODEL_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "LIMIT=$LIMIT"
echo "Output: $OUT_JSON"
echo "Log: $LOG_FILE"
echo "========================================================"

# vLLM backend with GPTQ INT8 quantization
# Note: Using single GPU, so tensor_parallel_size=1
BATCH_SIZE="${BATCH_SIZE:-auto}"

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_model_len=4096,quantization=gptq,dtype=auto,trust_remote_code=True"
  --tasks ceval-valid
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON"
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
