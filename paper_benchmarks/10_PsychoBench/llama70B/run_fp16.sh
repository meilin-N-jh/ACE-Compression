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

# Run PsychoBench for Llama70B FP16 via vLLM OpenAI API

set -euo pipefail

PB_DIR="${ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark"
SCRIPT="$PB_DIR/llama70B/run_psychobench_llama70b.py"
PORT="8000"
MODEL_NAME="llama31-70b-fp16"
RUN_NAME="Llama-70B-FP16"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

QUESTIONNAIRES="Empathy,BFI,BSRI,EPQ-R,LMS,DTDD,ECR-R,GSE,ICB,LOT-R,EIS,WLEIS,CABIN,16P"

RESULTS_DIR="${RESULTS_DIR:-$PB_DIR/llama70B/results}"
LOG_DIR="$PB_DIR/llama70B/logs"
RUN_LOG="$LOG_DIR/psychobench_fp16_${TIMESTAMP}.log"
MODEL_RESULTS_DIR="$RESULTS_DIR/$RUN_NAME"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

if [[ -d "$MODEL_RESULTS_DIR" ]]; then
  BACKUP_DIR="${MODEL_RESULTS_DIR}_backup_${TIMESTAMP}"
  mv "$MODEL_RESULTS_DIR" "$BACKUP_DIR"
  echo "[INFO] Existing results moved to: $BACKUP_DIR"
fi

if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
  echo "[ERROR] vLLM API server not found on port $PORT"
  echo "Please start the server first:"
  echo "  bash ${ARTIFACT_ROOT}/models/Llama3-70b/start_vllm_fp16.sh"
  exit 1
fi

echo "[INFO] Running PsychoBench..."
PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
  --model "$MODEL_NAME" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --questionnaire "$QUESTIONNAIRES" \
  --shuffle-count 1 \
  --test-count 10 \
  --name-exp "$RUN_NAME" \
  --significance-level 0.01 \
  --mode auto 2>&1 | tee -a "$RUN_LOG"

echo "[INFO] Results saved to: $MODEL_RESULTS_DIR"
echo "[INFO] Done. Logs: $RUN_LOG"
