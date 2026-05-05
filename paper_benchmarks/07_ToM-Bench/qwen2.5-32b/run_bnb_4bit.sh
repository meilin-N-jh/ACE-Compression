#!/bin/bash
# Run ToM-Bench for Qwen2.5-32B-BNB-4bit

python3 run_tombench_qwen25_32b_vllm.py \
    --model-display-name "Qwen2.5-32B-BNB-4bit" \
    --base-url "http://127.0.0.1:8201" \
    --model-name "qwen2.5-32b-bnb-4bit" \
    --language zh \
    --try-times 5 \
    --seed 42
