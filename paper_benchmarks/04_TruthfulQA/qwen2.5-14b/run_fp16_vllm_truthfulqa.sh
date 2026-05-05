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
# Description: Evaluate Qwen2.5-14B-Instruct (FP16) on TruthfulQA
# Env: qwen2.5 (conda)
# GPU: GPU 0
# Output: paper_benchmarks/04_TruthfulQA/launchers/qwen2.5-14b/results/
#
# Usage:
#   # 0-shot all tasks (mc1, mc2, gen)
#   ./run_fp16_vllm_truthfulqa.sh all
#   ./run_fp16_vllm_truthfulqa.sh mc
#   ./run_fp16_vllm_truthfulqa.sh mc1
#   ./run_fp16_vllm_truthfulqa.sh mc2
#   ./run_fp16_vllm_truthfulqa.sh gen
#
#   # 3-shot MC tasks
#   NUM_FEWSHOT=3 ./run_fp16_vllm_truthfulqa.sh mc
#   NUM_FEWSHOT=3 ./run_fp16_vllm_truthfulqa.sh mc1
#   NUM_FEWSHOT=3 ./run_fp16_vllm_truthfulqa.sh mc2
#
#   # Full evaluation (all 817 samples)
#   LIMIT=0 ./run_fp16_vllm_truthfulqa.sh all
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/Qwen2.5-14B/Qwen2.5-14B-Instruct-FP16"
OUT_DIR="$BASE/paper_benchmarks/04_TruthfulQA/launchers/qwen2.5-14b/results"
LOG_DIR="$BASE/paper_benchmarks/04_TruthfulQA/launchers/qwen2.5-14b/logs"

# Task type: all, mc, mc1, mc2, gen
TASK_TYPE="${1:-all}"

# Number of few-shot examples: 0 or 3
NUM_FEWSHOT="${NUM_FEWSHOT:-0}"

# Random seeds: format is <python>,<numpy>,<torch>,<fewshot>
SEED="${SEED:-0,1234,1234,1234}"

# LIMIT=10 for smoke test; set LIMIT=0 to run full.
LIMIT="${LIMIT:-10}"

# Use 1 GPU (GPU 0)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Model weights are local; keep transformers offline but allow BLEURT download.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR"
mkdir -p "$LOG_DIR"

# Determine task suffix based on NUM_FEWSHOT
if [[ "$NUM_FEWSHOT" == "0" ]]; then
    SHOT_SUFFIX="_0shot"
elif [[ "$NUM_FEWSHOT" == "3" ]]; then
    SHOT_SUFFIX="_3shot"
else
    echo "Error: NUM_FEWSHOT must be 0 or 3, got: $NUM_FEWSHOT"
    exit 1
fi

# Determine task name and parameters based on TASK_TYPE
case "$TASK_TYPE" in
    all)
        LM_EVAL_TASK="truthfulqa_mc2${SHOT_SUFFIX},truthfulqa_mc1${SHOT_SUFFIX},truthfulqa_gen"
        TASK_NAME="TruthfulQA-All"
        APPLY_CHAT_TEMPLATE="--apply_chat_template"
        GEN_KWARGS='--gen_kwargs temperature=0.0,do_sample=False,max_gen_toks=100'
        ;;
    mc)
        LM_EVAL_TASK="truthfulqa_mc2${SHOT_SUFFIX},truthfulqa_mc1${SHOT_SUFFIX}"
        TASK_NAME="TruthfulQA-MC"
        APPLY_CHAT_TEMPLATE=""
        GEN_KWARGS=""
        ;;
    mc2)
        LM_EVAL_TASK="truthfulqa_mc2${SHOT_SUFFIX}"
        TASK_NAME="TruthfulQA-MC2"
        APPLY_CHAT_TEMPLATE=""
        GEN_KWARGS=""
        ;;
    mc1)
        LM_EVAL_TASK="truthfulqa_mc1${SHOT_SUFFIX}"
        TASK_NAME="TruthfulQA-MC1"
        APPLY_CHAT_TEMPLATE=""
        GEN_KWARGS=""
        ;;
    gen)
        LM_EVAL_TASK="truthfulqa_gen"
        TASK_NAME="TruthfulQA-Gen"
        APPLY_CHAT_TEMPLATE="--apply_chat_template"
        GEN_KWARGS='--gen_kwargs temperature=0.0,do_sample=False,max_gen_toks=100'
        ;;
    *)
        echo "Error: Invalid TASK_TYPE '$TASK_TYPE'. Must be 'all', 'mc', 'mc2', 'mc1', or 'gen'"
        echo "  all - run mc1, mc2, and gen (recommended for 0-shot)"
        echo "  mc  - run both mc1 and mc2"
        echo "  mc2 - run only mc2"
        echo "  mc1 - run only mc1"
        echo "  gen - run only gen"
        exit 1
        ;;
esac

# Build output filename (JSON without timestamp, log with timestamp)
TS="$(date +%F_%H%M%S)"
SHOT_LABEL="${NUM_FEWSHOT}shot"
OUT_JSON="$OUT_DIR/truthfulqa_fp16_${TASK_TYPE}_${SHOT_LABEL}.json"
LOG_FILE="$LOG_DIR/truthfulqa_fp16_${TASK_TYPE}_${SHOT_LABEL}_${TS}.log"

# Send *all* output to the log and console.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting $TASK_NAME Evaluation"
echo "Model: Qwen2.5-14B-Instruct FP16"
echo "Date: $(date)"
echo "========================================================"
echo "Configuration:"
echo "  Task: $LM_EVAL_TASK"
echo "  Few-shot: $NUM_FEWSHOT"
echo "  Seed: $SEED"
echo "  Limit: $LIMIT"
echo "  Model: $MODEL_DIR"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Output: $OUT_JSON"
echo "  Log: $LOG_FILE"
echo "========================================================"

# vLLM backend arguments
BATCH_SIZE="${BATCH_SIZE:-auto}"

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=4096,dtype=auto,trust_remote_code=True,max_num_seqs=256,max_num_batched_tokens=8192"
  --tasks "$LM_EVAL_TASK"
  --seed "$SEED"
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON"
  --log_samples
  --verbosity INFO
)

# Add task-specific arguments
if [[ -n "$APPLY_CHAT_TEMPLATE" ]]; then
    args+=($APPLY_CHAT_TEMPLATE)
fi

if [[ -n "$GEN_KWARGS" ]]; then
    args+=($GEN_KWARGS)
fi

# Add limit if specified
if [[ "$LIMIT" != "0" && "$LIMIT" != "" ]]; then
  args+=(--limit "$LIMIT")
fi

echo "Starting evaluation..."
echo "Command: lm-eval" "${args[@]}"
echo "========================================================"

time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args[@]}"

echo "========================================================"
echo "Evaluation Finished."
echo "Results saved to: $OUT_JSON"
echo "Log saved to: $LOG_FILE"
echo "========================================================"
