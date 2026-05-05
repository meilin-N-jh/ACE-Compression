#!/bin/bash
# Run ToM-Bench for Qwen2.5-14B-BNB-4bit

python3 run_tombench_qwen25_vllm.py \
    --model-display-name "Qwen2.5-14B-BNB-4bit" \
    --base-url "http://127.0.0.1:8101" \
    --model-name "qwen2.5-14b-bnb-4bit" \
    --language zh \
    --try-times 5 \
    --seed 42
