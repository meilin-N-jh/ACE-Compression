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


EQBENCH_DIR="${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark"
CONFIG_FILE="${EQBENCH_CONFIG:-$EQBENCH_DIR/qwen2.5-32b/config/config_fp16.cfg}"
PORT="8200"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

RESULTS_DIR="${RESULTS_DIR:-$EQBENCH_DIR/qwen2.5-32b/results}"
RUN_LOG="$EQBENCH_DIR/qwen2.5-32b/logs/eqbench_fp16_${TIMESTAMP}.log"

mkdir -p "$EQBENCH_DIR/qwen2.5-32b/logs"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "EQ-Bench: Qwen2.5-32B FP16"
echo "=========================================="
echo "API Port: $PORT"
echo "=========================================="
echo ""

echo "[INFO] Checking vLLM API server on port $PORT..."
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "[ERROR] vLLM API server not found on port $PORT"
    echo "Please start the server first:"
    echo "  bash ${ARTIFACT_ROOT}/models/Qwen2.5-32B/start_vllm_fp16.sh"
    exit 1
fi

echo "[INFO] API server is ready!"
echo ""


echo "[INFO] Running EQ-Bench..."
cd "$EQBENCH_DIR"
EQBENCH_OUTPUT_DIR="$RESULTS_DIR" PYTHONUNBUFFERED=1 python -u eq-bench.py --config "$CONFIG_FILE" 2>&1 | tee -a "$RUN_LOG"

echo ""
echo "=========================================="
echo "Done. Logs: $RUN_LOG"
echo "=========================================="
