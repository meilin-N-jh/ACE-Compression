#!/bin/bash
# Run ToM-Bench for Qwen2.5-14B Unstructured Sparse

python3 run_tombench_qwen25_vllm.py \
    --model-display-name "Qwen2.5-14B-Unstructured-Sparse" \
    --base-url "http://127.0.0.1:8106" \
    --model-name "qwen2.5-14b-unstructured-sparse" \
    --language zh \
    --try-times 5 \
    --seed 42
