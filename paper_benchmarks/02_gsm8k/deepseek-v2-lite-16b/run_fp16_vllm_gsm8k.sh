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
# Script: run_fp16_vllm_gsm8k.sh
# Description: Evaluate DeepSeek-V2-Lite-16B (FP16) on GSM8K
# -----------------------------------------------------------------------------

BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat"
OUT_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/deepseek-v2-lite-16b/results"
LOG_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/deepseek-v2-lite-16b/logs"

LIMIT="${LIMIT:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/gsm8k_cot_fp16_vllm_${TS}.json"
LOG_FILE="$LOG_DIR/gsm8k_cot_fp16_vllm_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting GSM8K - DeepSeek-V2-Lite-16B (FP16)"
echo "========================================================"

args=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.90,max_model_len=4096,dtype=auto,trust_remote_code=True,max_num_seqs=1024,max_num_batched_tokens=32768"
  --tasks gsm8k_cot
  --apply_chat_template
  --fewshot_as_multiturn
  --batch_size auto
  --output_path "$OUT_JSON"
  --log_samples
  --verbosity INFO
)

[[ "$LIMIT" != "0" ]] && args+=(--limit "$LIMIT")

time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args[@]}"

echo "Done!"
