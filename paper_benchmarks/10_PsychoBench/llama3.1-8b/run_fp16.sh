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

# Run PsychoBench for Llama3.1-8b FP16 via vLLM OpenAI API
# 需要先启动vLLM服务器

set -euo pipefail

PB_DIR="${ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark"
SCRIPT="$PB_DIR/llama3.1-8b/run_psychobench_llama.py"
PORT="8400"
MODEL_NAME="meta-llama-3.1-8b-instruct-fp16"
RUN_NAME="Llama3.1-8B-FP16"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

QUESTIONNAIRES="Empathy,BFI,BSRI,EPQ-R,LMS,DTDD,ECR-R,GSE,ICB,LOT-R,EIS,WLEIS,CABIN,16P"

RESULTS_DIR="${RESULTS_DIR:-$PB_DIR/llama3.1-8b/results}"
LOG_DIR="$PB_DIR/llama3.1-8b/logs"
RUN_LOG="$LOG_DIR/psychobench_fp16_${TIMESTAMP}.log"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# 检查API服务器是否已启动
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
  echo "[ERROR] vLLM API server not found on port $PORT"
  echo "Please start the server first:"
  echo "  bash ${ARTIFACT_ROOT}/models/llama3.1-8b/start_vllm_fp16.sh"
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

# 移动结果
SRC_RESULTS="$PB_DIR/results/$RUN_NAME"
if [[ -d "$SRC_RESULTS" ]]; then
  DEST_RESULTS="$RESULTS_DIR/$RUN_NAME"
  if [[ -d "$DEST_RESULTS" ]]; then
    DEST_RESULTS="$RESULTS_DIR/${RUN_NAME}_${TIMESTAMP}"
  fi
  mv "$SRC_RESULTS" "$DEST_RESULTS"
  echo "[INFO] Results saved to: $DEST_RESULTS"
fi

echo "[INFO] Done. Logs: $RUN_LOG"
