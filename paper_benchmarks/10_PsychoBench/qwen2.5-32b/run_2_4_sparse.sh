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

# Run PsychoBench for Qwen2.5-32B 2:4 Sparse via vLLM OpenAI API

set -euo pipefail

PB_DIR="${ARTIFACT_ROOT}/paper_benchmarks/10_PsychoBench/benchmark"
SCRIPT="$PB_DIR/qwen2.5-32b/run_psychobench_qwen25_32b.py"
PORT="8208"
MODEL_NAME="qwen2.5-32b-2to4-sparse"
RUN_NAME="Qwen2.5-32B-2:4-Sparse"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

QUESTIONNAIRES="Empathy,BFI,BSRI,EPQ-R,LMS,DTDD,ECR-R,GSE,ICB,LOT-R,EIS,WLEIS,CABIN,16P"

RESULTS_DIR="${RESULTS_DIR:-$PB_DIR/qwen2.5-32b/results}"
LOG_DIR="$PB_DIR/qwen2.5-32b/logs"
RUN_LOG="$LOG_DIR/psychobench_2_4_sparse_${TIMESTAMP}.log"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# 检查API服务器是否已启动
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
  echo "[ERROR] vLLM API server not found on port $PORT"
  echo "Please start the server first:"
  echo "  bash ${ARTIFACT_ROOT}/models/Qwen2.5-32B/start_vllm_2_4_sparse.sh"
  exit 1
fi

echo "[INFO] Running PsychoBench..."
PYTHONUNBUFFERED=1 python -u "$SCRIPT" \
  --model "$MODEL_NAME" \
  --base-url "http://127.0.0.1:${PORT}" \
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

# 移动 figures
FIG_DIR="$PB_DIR/results/figures"
if [[ -d "$FIG_DIR" ]]; then
  mkdir -p "$RESULTS_DIR/figures"
  find "$FIG_DIR" -maxdepth 1 -type f -name "${RUN_NAME}-*.png" -exec mv {} "$RESULTS_DIR/figures/" \; 2>/dev/null || true
fi

# 移动 prompts/responses
if [[ -d "$PB_DIR/prompts" ]]; then
  mkdir -p "$RESULTS_DIR/prompts"
  find "$PB_DIR/prompts" -maxdepth 1 -type f -name "${RUN_NAME}-*.txt" -exec mv {} "$RESULTS_DIR/prompts/" \; 2>/dev/null || true
fi
if [[ -d "$PB_DIR/responses" ]]; then
  mkdir -p "$RESULTS_DIR/responses"
  find "$PB_DIR/responses" -maxdepth 1 -type f -name "${RUN_NAME}-*.txt" -exec mv {} "$RESULTS_DIR/responses/" \; 2>/dev/null || true
fi

echo "[INFO] Done. Logs: $RUN_LOG"
