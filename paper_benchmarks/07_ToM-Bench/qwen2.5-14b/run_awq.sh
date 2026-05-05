#!/bin/bash
# Run ToM-Bench for Qwen2.5-14B-AWQ
python3 run_tombench_qwen25_vllm.py \
    --model-display-name "Qwen2.5-14B-AWQ" \
    --base-url "http://127.0.0.1:8102" \
    --model-name "qwen2.5-14b-awq" \
    --language zh \
    --try-times 5 \
    --seed 42
