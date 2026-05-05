#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${1:-$ROOT_DIR/runs}"

if [[ ! -d "$RUNS_DIR" ]]; then
  echo "[ERROR] Runs directory not found: $RUNS_DIR"
  exit 1
fi

cleanup_server() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi
}

wait_for_server() {
  local port="$1"
  local timeout_s="${2:-180}"
  local waited=0
  while (( waited < timeout_s )); do
    if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
  done
  return 1
}

for run_dir in "$RUNS_DIR"/*; do
  [[ -d "$run_dir" ]] || continue
  [[ -f "$run_dir/run.env" ]] || { echo "[WARN] skip (missing run.env): $run_dir"; continue; }

  unset RUN_NAME MODEL_PATH MODEL_ID PORT GPU VLLM_ARGS
  # shellcheck disable=SC1090
  source "$run_dir/run.env"

  : "${RUN_NAME:?RUN_NAME missing in run.env}"
  : "${MODEL_PATH:?MODEL_PATH missing in run.env}"
  : "${MODEL_ID:?MODEL_ID missing in run.env}"
  : "${PORT:?PORT missing in run.env}"

  ts="$(date +%Y-%m-%d_%H%M%S)"
  server_log="$run_dir/logs/vllm_${ts}.log"
  bench_log="$run_dir/logs/eqbench_${ts}.log"

  echo "=========================================="
  echo "[RUN] $RUN_NAME"
  echo "Model path: $MODEL_PATH"
  echo "Model id:   $MODEL_ID"
  echo "Port:       $PORT"
  echo "=========================================="

  read -r -a extra_args <<< "${VLLM_ARGS:-}"
  vllm_cmd=(python -m vllm.entrypoints.openai.api_server
    --host 127.0.0.1
    --port "$PORT"
    --model "$MODEL_PATH"
    --served-model-name "$MODEL_ID"
  )
  vllm_cmd+=("${extra_args[@]}")

  if [[ -n "${GPU:-}" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "${vllm_cmd[@]}" >"$server_log" 2>&1 &
  else
    "${vllm_cmd[@]}" >"$server_log" 2>&1 &
  fi
  vllm_pid=$!

  if ! wait_for_server "$PORT" 240; then
    echo "[ERROR] vLLM failed to become ready for $RUN_NAME"
    cleanup_server "$vllm_pid"
    exit 1
  fi

  set +e
  (
    cd "$ROOT_DIR"
    EQBENCH_OUTPUT_DIR="$run_dir/results" \
      PYTHONUNBUFFERED=1 \
      python -u "$ROOT_DIR/eq-bench.py" --config "$run_dir/config.cfg"
  ) 2>&1 | tee "$bench_log"
  eq_exit=${PIPESTATUS[0]}
  set -e

  cleanup_server "$vllm_pid"

  if [[ $eq_exit -ne 0 ]]; then
    echo "[ERROR] EQ-Bench failed for $RUN_NAME (exit=$eq_exit)"
    exit $eq_exit
  fi

  echo "[OK] Completed $RUN_NAME"
  echo "Logs: $bench_log"
  echo

done

echo "All runs completed."
