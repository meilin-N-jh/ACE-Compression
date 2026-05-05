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

# Run RoleEval for Llama3.1-8b FP16
# 需要先启动 vLLM 服务

cd ${ARTIFACT_ROOT}/paper_benchmarks/06_roleeval/benchmark/llama3.1-8b

# 运行所有模型
python run_roleeval_llama.py --model all --lang both --subset both --split test

echo "RoleEval 完成！"
