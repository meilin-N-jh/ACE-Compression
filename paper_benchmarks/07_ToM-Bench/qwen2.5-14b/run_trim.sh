#!/bin/bash
# Run ToM-Bench for Qwen2.5-14B-Trim
python3 run_tombench_qwen25_vllm.py \
    --model-display-name "Qwen2.5-14B-Trim" \
    --base-url "http://127.0.0.1:8104" \
    --model-name "qwen2.5-14b-trim" \
    --language zh \
    --try-times 5 \
    --seed 42
