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
# Script: run_fp16_vllm_truthfulqa.sh
# Description: Evaluate Meta-Llama-3.1-8B-Instruct (FP16) on TruthfulQA with BLEURT
#              Only runs 0-shot generation task (truthfulqa_gen)
# Env: qwen2.5 (conda)
# GPU: 4
# Output: ${ARTIFACT_ROOT}/paper_benchmarks/04_TruthfulQA/launchers/llama3.1-8b/results
#
# Usage:
#   # Default: run 0-shot gen with BLEURT (LIMIT=10 for smoke test)
#   ./run_fp16_vllm_truthfulqa.sh
#
#   # Full evaluation (all 817 samples)
#   LIMIT=0 ./run_fp16_vllm_truthfulqa.sh
#
# Note: BLEURT is computed automatically after lm-eval completes
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct"
OUT_DIR="${ARTIFACT_ROOT}/paper_benchmarks/04_TruthfulQA/launchers/llama3.1-8b/results"
LOG_DIR="${ARTIFACT_ROOT}/paper_benchmarks/04_TruthfulQA/launchers/llama3.1-8b/logs"

# BLEURT checkpoint (default: bleurt-large-128)
BLEURT_CHECKPOINT="${BLEURT_CHECKPOINT:-bleurt-large-128}"

# Task: only gen (0-shot generation)
LM_EVAL_TASK="truthfulqa_gen"
SHOT_SUFFIX="_0shot"

# Random seeds: format is <python>,<numpy>,<torch>,<fewshot>
SEED="${SEED:-0,1234,1234,1234}"

# LIMIT=10 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-10}"

# Use GPU 4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# Model weights are local; keep transformers offline.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

# Build output filename
TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/truthfulqa_fp16_gen_0shot.json"
LOG_FILE="$LOG_DIR/truthfulqa_fp16_gen_0shot_${TS}.log"

# Send *all* output to the log and console.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting TruthfulQA Gen 0-shot Evaluation with BLEURT"
echo "Model: Meta-Llama-3.1-8B-Instruct (FP16)"
echo "Date: $(date)"
echo "========================================================"
echo "Configuration:"
echo "  Task: $LM_EVAL_TASK"
echo "  Few-shot: 0"
echo "  Seed: $SEED"
echo "  Limit: $LIMIT"
echo "  Model: $MODEL_DIR"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  BLEURT Checkpoint: $BLEURT_CHECKPOINT"
echo "  Output: $OUT_JSON"
echo "  Log: $LOG_FILE"
echo "========================================================"

# vLLM backend arguments
BATCH_SIZE="${BATCH_SIZE:-auto}"

# Tensor parallel size
TP_SIZE=1

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=$TP_SIZE,gpu_memory_utilization=0.90,max_model_len=4096,dtype=auto,trust_remote_code=True,max_num_seqs=256,max_num_batched_tokens=8192"
  --tasks "$LM_EVAL_TASK"
  --seed "$SEED"
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON"
  --log_samples
  --verbosity INFO
  --apply_chat_template
  --gen_kwargs "temperature=0.0,do_sample=False,max_gen_toks=100"
)

# Add limit if specified
if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args+=(--limit "$LIMIT")
fi

echo "Starting lm-eval evaluation..."
echo "Command: lm-eval" "${args[@]}"
echo "========================================================"

time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args[@]}"

echo "========================================================"
echo "lm-eval Finished."
echo "========================================================"

# ========================================================
# Compute BLEURT for generation task
# ========================================================
echo ""
echo "========================================================"
echo "Results saved to: $OUT_JSON"
echo "Log saved to: $LOG_FILE"
echo "========================================================"
