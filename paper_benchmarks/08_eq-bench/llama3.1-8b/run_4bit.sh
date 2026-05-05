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

# Run EQ-Bench for Llama3.1-8b BNB-4bit via vLLM OpenAI API
# 需要先启动vLLM服务器

set -euo pipefail

EQBENCH_DIR="${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark"
CONFIG_FILE="${EQBENCH_CONFIG:-$EQBENCH_DIR/llama3.1-8b/config/config_4bit.cfg}"
PORT="8401"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

RESULTS_DIR="${RESULTS_DIR:-$EQBENCH_DIR/llama3.1-8b/results}"
RUN_LOG="$EQBENCH_DIR/llama3.1-8b/logs/eqbench_4bit_${TIMESTAMP}.log"

mkdir -p "$EQBENCH_DIR/llama3.1-8b/logs"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "EQ-Bench: Llama3.1-8B BNB-4bit"
echo "=========================================="

# 检查API服务器是否已启动
if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "[ERROR] vLLM API server not found on port $PORT"
    echo "Please start the server first:"
    echo "  bash ${ARTIFACT_ROOT}/models/llama3.1-8b/start_vllm_4bit.sh"
    exit 1
fi

echo "[INFO] API server is ready!"

# Backup and switch config

# Run EQ-Bench
echo "[INFO] Running EQ-Bench..."
cd "$EQBENCH_DIR"
EQBENCH_OUTPUT_DIR="$RESULTS_DIR" PYTHONUNBUFFERED=1 python -u eq-bench.py --config "$CONFIG_FILE" 2>&1 | tee -a "$RUN_LOG"

echo "Done!"
