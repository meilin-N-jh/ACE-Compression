#!/usr/bin/env bash
# Run all TruthfulQA 0-shot gen evaluations for deepseek-v2-lite-16b
# GPU: 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "Running all TruthfulQA 0-shot gen for deepseek-v2-lite-16b"
echo "GPU: 3"
echo "========================================================"


echo ""
echo "========================================================"
echo "Running fp16..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=3 LIMIT=0 bash "$SCRIPT_DIR/run_fp16_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running awq..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=3 LIMIT=0 bash "$SCRIPT_DIR/run_awq_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running bnb_4bit..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=3 LIMIT=0 bash "$SCRIPT_DIR/run_bnb_4bit_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running 2_4_sparse..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=3 LIMIT=0 bash "$SCRIPT_DIR/run_2_4_sparse_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running unstructured_sparse..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=3 LIMIT=0 bash "$SCRIPT_DIR/run_unstructured_sparse_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "All evaluations for deepseek-v2-lite-16b completed!"
echo "========================================================"
