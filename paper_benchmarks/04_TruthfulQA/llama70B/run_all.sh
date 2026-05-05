#!/usr/bin/env bash
# Run all TruthfulQA 0-shot gen evaluations for llama70B
# GPU: 5,6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "Running all TruthfulQA 0-shot gen for llama70B"
echo "GPU: 5,6"
echo "========================================================"


echo ""
echo "========================================================"
echo "Running fp16..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=5,6 LIMIT=0 bash "$SCRIPT_DIR/run_fp16_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running awq..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=5,6 LIMIT=0 bash "$SCRIPT_DIR/run_awq_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running gptq_int4..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=5,6 LIMIT=0 bash "$SCRIPT_DIR/run_gptq_int4_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running gptq_int8..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=5,6 LIMIT=0 bash "$SCRIPT_DIR/run_gptq_int8_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running 4bit..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=5,6 LIMIT=0 bash "$SCRIPT_DIR/run_4bit_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running 2_4_sparse..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=5,6 LIMIT=0 bash "$SCRIPT_DIR/run_2_4_sparse_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running unstructured_sparse..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=5,6 LIMIT=0 bash "$SCRIPT_DIR/run_unstructured_sparse_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "All evaluations for llama70B completed!"
echo "========================================================"
