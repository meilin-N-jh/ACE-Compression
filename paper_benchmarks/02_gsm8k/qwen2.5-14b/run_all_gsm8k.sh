#!/bin/bash
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
# Script: run_all_gsm8k.sh
# Description: Run GSM8K evaluation for all 5 Qwen2.5-14B models in parallel
# Usage: ./run_all_gsm8k.sh
# Notes:
#   - Runs all 5 models simultaneously on GPUs 0-4
#   - Full evaluation (LIMIT=0)
#   - Logs saved to logs/ directory
# -----------------------------------------------------------------------------

cd ${ARTIFACT_ROOT}/paper_benchmarks/02_gsm8k/launchers/qwen2.5-14b

echo "========================================================"
echo "Starting GSM8K evaluation for all Qwen2.5-14B models"
echo "Date: $(date)"
echo "========================================================"
echo ""

# Run all 5 models in parallel with LIMIT=0 for full evaluation
LIMIT=0 ./run_fp16_vllm_gsm8k.sh &
PID_FP16=$!

LIMIT=0 ./run_bnb_4bit_vllm_gsm8k.sh &
PID_BNB=$!

LIMIT=0 ./run_awq_vllm_gsm8k.sh &
PID_AWQ=$!

LIMIT=0 ./run_gptq_int4_vllm_gsm8k.sh &
PID_GPTQ=$!

LIMIT=0 ./run_trim_vllm_gsm8k.sh &
PID_TRIM=$!

echo "All evaluations started in parallel:"
echo "  - FP16 (PID: $PID_FP16) on GPU 0"
echo "  - BNB 4bit (PID: $PID_BNB) on GPU 1"
echo "  - AWQ (PID: $PID_AWQ) on GPU 2"
echo "  - GPTQ INT4 (PID: $PID_GPTQ) on GPU 3"
echo "  - Trim (PID: $PID_TRIM) on GPU 4"
echo ""
echo "Monitor logs with: tail -f logs/*.log"
echo "========================================================"

# Wait for all processes to complete
wait $PID_FP16
wait $PID_BNB
wait $PID_AWQ
wait $PID_GPTQ
wait $PID_TRIM

echo ""
echo "========================================================"
echo "All GSM8K evaluations completed!"
echo "Date: $(date)"
echo "Results saved in: results/"
echo "========================================================"
