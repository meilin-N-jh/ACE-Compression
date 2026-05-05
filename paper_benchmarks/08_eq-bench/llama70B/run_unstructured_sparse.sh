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

# Run EQ-Bench for Llama70B Unstructured Sparse via vLLM OpenAI API

set -euo pipefail

EQBENCH_DIR="${ARTIFACT_ROOT}/paper_benchmarks/08_eq-bench/benchmark"
CONFIG_FILE="${EQBENCH_CONFIG:-$EQBENCH_DIR/llama70B/config/config_unstructured_sparse.cfg}"
PORT="8011"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)

RESULTS_DIR="${RESULTS_DIR:-$EQBENCH_DIR/llama70B/results}"
RUN_LOG="$EQBENCH_DIR/llama70B/logs/eqbench_unstructured_sparse_${TIMESTAMP}.log"

mkdir -p "$EQBENCH_DIR/llama70B/logs"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "EQ-Bench: Llama70B Unstructured Sparse"
echo "=========================================="

if ! curl -s "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    echo "[ERROR] vLLM API server not found on port $PORT"
    exit 1
fi


cd "$EQBENCH_DIR"
EQBENCH_OUTPUT_DIR="$RESULTS_DIR" PYTHONUNBUFFERED=1 python -u eq-bench.py --config "$CONFIG_FILE" 2>&1 | tee -a "$RUN_LOG"


echo "Done. Logs: $RUN_LOG"
