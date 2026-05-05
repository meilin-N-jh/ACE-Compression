#!/bin/bash
# Run ToM-Bench for Qwen2.5-7B 2:4 Sparse

python3 run_tombench_qwen25_vllm.py \
    --model-display-name "Qwen2.5-7B-2:4-Sparse" \
    --base-url "http://127.0.0.1:8404" \
    --model-name "qwen2.5-7b-2to4-sparse" \
    --language zh \
    --try-times 5 \
    --seed 42
