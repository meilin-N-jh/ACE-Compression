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

# Run EQ-Bench for Qwen2.5-14B FP16 via vLLM OpenAI API
# 需要先手动启动vLLM服务器，并在终端激活环境: conda activate benchmark

set -euo pipefail

EQBENCH_DIR="${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark"
CONFIG_FILE="${EQBENCH_CONFIG:-$EQBENCH_DIR/qwen2.5-14b/config/config_fp16.cfg}"
PORT="8100"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

RESULTS_DIR="${RESULTS_DIR:-$EQBENCH_DIR/qwen2.5-14b/results}"
RUN_LOG="$EQBENCH_DIR/qwen2.5-14b/logs/eqbench_fp16_${TIMESTAMP}.log"

mkdir -p "$EQBENCH_DIR/qwen2.5-14b/logs"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "EQ-Bench: Qwen2.5-14B FP16"
echo "=========================================="
echo "API Port: $PORT"
echo "=========================================="
echo ""

# 检查API服务器是否已启动
echo "[INFO] Checking vLLM API server on port $PORT..."
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "[ERROR] vLLM API server not found on port $PORT"
    echo "Please start the server first:"
    echo "  bash ${ARTIFACT_ROOT}/models/Qwen2.5-14B/start_vllm_fp16.sh"
    exit 1
fi

echo "[INFO] API server is ready!"
echo ""

# Backup and switch config

# Run EQ-Bench
echo "[INFO] Running EQ-Bench..."
cd "$EQBENCH_DIR"
EQBENCH_OUTPUT_DIR="$RESULTS_DIR" PYTHONUNBUFFERED=1 python -u eq-bench.py --config "$CONFIG_FILE" 2>&1 | tee -a "$RUN_LOG"

echo "Done. Logs: $RUN_LOG"
echo "=========================================="
