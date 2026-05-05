#!/usr/bin/env bash
# Run all TruthfulQA 0-shot gen evaluations for qwen2.5-14b
# GPU: 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "Running all TruthfulQA 0-shot gen for qwen2.5-14b"
echo "GPU: 1"
echo "========================================================"


echo ""
echo "========================================================"
echo "Running fp16..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=1 LIMIT=0 bash "$SCRIPT_DIR/run_fp16_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running awq..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=1 LIMIT=0 bash "$SCRIPT_DIR/run_awq_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running gptq_int4..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=1 LIMIT=0 bash "$SCRIPT_DIR/run_gptq_int4_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running bnb_4bit..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=1 LIMIT=0 bash "$SCRIPT_DIR/run_bnb_4bit_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running trim..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=1 LIMIT=0 bash "$SCRIPT_DIR/run_trim_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running 2_4_sparse..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=1 LIMIT=0 bash "$SCRIPT_DIR/run_2_4_sparse_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "Running unstructured_sparse..."
echo "========================================================"
CUDA_VISIBLE_DEVICES=1 LIMIT=0 bash "$SCRIPT_DIR/run_unstructured_sparse_vllm_truthfulqa.sh" gen

echo ""
echo "========================================================"
echo "All evaluations for qwen2.5-14b completed!"
echo "========================================================"
