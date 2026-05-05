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

# Run PsychoBench for Qwen2.5-7B GPTQ-INT4 via vLLM OpenAI API
# 需要先手动启动vLLM服务器，并在终端激活环境: conda activate qwen2.5

set -euo pipefail

PB_DIR="${ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark"
SCRIPT="$PB_DIR/qwen2.5-7b/run_psychobench_qwen25_7b.py"
PORT="8403"
MODEL_NAME="qwen2.5-7b-gptq-int4"
RUN_NAME="Qwen2.5-7B-GPTQ-INT4"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

QUESTIONNAIRES="Empathy,BFI,BSRI,EPQ-R,LMS,DTDD,ECR-R,GSE,ICB,LOT-R,EIS,WLEIS,CABIN,16P"

RESULTS_DIR="${RESULTS_DIR:-$PB_DIR/qwen2.5-7b/results}"
LOG_DIR="$PB_DIR/qwen2.5-7b/logs"
RUN_LOG="$LOG_DIR/psychobench_gptq_int4_${TIMESTAMP}.log"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# 检查API服务器是否已启动
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
  echo "[ERROR] vLLM API server not found on port $PORT"
  echo "Please start the server first:"
  echo "  bash ${ARTIFACT_ROOT}/models/Qwen2.5-7B/start_vllm_gptq_int4.sh"
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
