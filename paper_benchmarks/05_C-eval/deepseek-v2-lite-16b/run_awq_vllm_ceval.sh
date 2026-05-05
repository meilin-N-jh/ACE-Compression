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


BASE="${ARTIFACT_ROOT}"
MODEL_DIR="$BASE/models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-AWQ"
OUT_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/deepseek-v2-lite-16b/results"
LOG_DIR="$BASE/paper_benchmarks/05_C-eval/launchers/deepseek-v2-lite-16b/logs"

LIMIT="${LIMIT:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
LOG_FILE="$LOG_DIR/ceval_awq_vllm_${TS}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "Starting C-Eval - DeepSeek-V2-Lite-16B (AWQ)"
echo "========================================================"

BATCH_SIZE="${BATCH_SIZE:-auto}"

# Valid
OUT_JSON_VALID="$OUT_DIR/ceval_valid_awq_vllm_${TS}.json"
args_valid=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.95,max_model_len=4096,dtype=auto,trust_remote_code=True"
  --tasks ceval-valid
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON_VALID"
  --verbosity INFO
)
[[ "$LIMIT" != "0" ]] && args_valid+=(--limit "$LIMIT")

time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args_valid[@]}"

# Test
OUT_JSON_TEST="$OUT_DIR/ceval_test_awq_vllm_${TS}.json"
args_test=(
  --model vllm
  --model_args "pretrained=$MODEL_DIR,tensor_parallel_size=1,gpu_memory_utilization=0.95,max_model_len=4096,dtype=auto,trust_remote_code=True"
  --include_path "$BASE/paper_benchmarks/05_C-eval/launchers/tasks/ceval-test"
  --tasks ceval-test
  --batch_size "$BATCH_SIZE"
  --output_path "$OUT_JSON_TEST"
  --verbosity INFO
)
[[ "$LIMIT" != "0" ]] && args_test+=(--limit "$LIMIT")

time conda run --no-capture-output -n qwen2.5 python -m lm_eval "${args_test[@]}"

echo "Done!"
