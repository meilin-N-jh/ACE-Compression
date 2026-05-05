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
MODEL_DIR="$BASE/models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-gptq-4bit"
OUT_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/deepseek-v2-lite-16b/results"
LOG_DIR="$BASE/paper_benchmarks/02_gsm8k/launchers/deepseek-v2-lite-16b/logs"
ENV_NAME="${ENV_NAME:-qwen2.5}"

# DeepSeek-V2-Lite GPTQ INT4 is often incompatible with vLLM kernels on non-Hopper GPUs.
# Default to HF backend for stability. Set ENGINE=vllm to force vLLM path.
ENGINE="${ENGINE:-hf}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-gptq_bitblas}"

LIMIT="${LIMIT:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_DIR" "$LOG_DIR"

TS="$(date +%F_%H%M%S)"
OUT_JSON="$OUT_DIR/gsm8k_cot_gptq_int4_${ENGINE}_${TS}.json"
LOG_FILE="$LOG_DIR/gsm8k_cot_gptq_int4_${ENGINE}_${TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "$ENGINE" == "vllm" ]]; then
  args=(
    --model vllm
    --model_args "pretrained=$MODEL_DIR,quantization=$VLLM_QUANTIZATION,enforce_eager=True,tensor_parallel_size=1,gpu_memory_utilization=0.45,max_model_len=4096,dtype=float16,trust_remote_code=True"
    --tasks gsm8k_cot
    --apply_chat_template
    --batch_size auto
    --output_path "$OUT_JSON"
    --log_samples
  )
else
  BATCH_SIZE="${BATCH_SIZE:-1}"
  args=(
    --model hf
    --model_args "pretrained=$MODEL_DIR,trust_remote_code=True,dtype=float16"
    --tasks gsm8k_cot
    --apply_chat_template
    --fewshot_as_multiturn
    --batch_size "$BATCH_SIZE"
    --output_path "$OUT_JSON"
    --log_samples
  )
fi

[[ "$LIMIT" != "0" ]] && args+=(--limit "$LIMIT")

echo "ENGINE=$ENGINE"
echo "VLLM_QUANTIZATION=$VLLM_QUANTIZATION"
echo "ENV_NAME=$ENV_NAME"
time conda run --no-capture-output -n "$ENV_NAME" python -m lm_eval "${args[@]}"

echo "Done!"
