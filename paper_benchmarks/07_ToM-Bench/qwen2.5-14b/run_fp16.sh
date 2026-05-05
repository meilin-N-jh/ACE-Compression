#!/bin/bash
# Run ToM-Bench for Qwen2.5-14B-FP16

python3 run_tombench_qwen25_vllm.py \
    --model-display-name "Qwen2.5-14B-FP16" \
    --base-url "http://127.0.0.1:8100" \
    --model-name "qwen2.5-14b-fp16" \
    --language zh \
    --try-times 5 \
    --seed 42
