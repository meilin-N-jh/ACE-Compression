# Compute Requirements Estimate

This document estimates the GPU compute required to reproduce the ACE-Compression benchmark evaluations.

## Experimental Matrix

| Model Family | Size | Variants | GPU Config |
|---|---:|---:|---|
| Qwen2.5-7B | 7B | 6 (FP16, BnB-4bit, AWQ, GPTQ-INT4, 2:4-Sparse, Unstructured-Sparse) | 1 GPU |
| Llama3.1-8B | 8B | 6 (same as above) | 1 GPU |
| Qwen2.5-14B | 14B | 7 (above + Trim) | 1 GPU |
| DeepSeek-V2-Lite-16B | 16B MoE, ~2.4B active | 5 | 1 GPU |
| Qwen2.5-32B | 32B | 6 | 1-2 GPUs |
| Llama3.1-70B | 70B | 7 (above + GPTQ-INT8) | 2 GPUs (TP=2) |
| **Total** | | **37 variants** | |

## Per-Benchmark Per-Variant Estimates

### Category A — Direct lm-eval Runs

| Benchmark | Samples | Est. output tok/sample | Total gen tokens | 7B/8B (1 GPU) | 14B/16B (1 GPU) | 32B (1-2 GPUs) | 70B (2 GPUs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| IFEval | 541 | ~200 | ~108K | ~3 min | ~5 min | ~10 min | ~15 min |
| GSM8K | 1,319 | ~300 (CoT) | ~396K | ~6 min | ~12 min | ~22 min | ~35 min |
| HumanEval (greedy) | 164 | ~200 | ~33K | ~2 min | ~3 min | ~5 min | ~8 min |
| HumanEval (pass@10, n=20) | 164 x 20 | ~200 | ~656K | ~10 min | ~18 min | ~35 min | ~55 min |
| HumanEval (pass@100, n=100) | 164 x 100 | ~200 | ~3.28M | ~60 min | ~90 min | ~200 min | ~300 min |
| TruthfulQA (0-shot) | 817 | ~50 | ~41K | ~10 min | ~20 min | ~30 min | ~40 min |
| TruthfulQA (3-shot x 3 seeds) | 817 x 3 | ~50 | ~123K | ~10 min | ~20 min | ~20 min | ~30 min |
| C-Eval (valid + test) | 2,736 | ~20 | ~55K | ~3 min | ~4 min | ~8 min | ~12 min |
| RoleEval (4 configs) | ~1,000 x 4 | ~5 | ~20K | ~5 min | ~8 min | ~12 min | ~18 min |

### Category B — vLLM API-Service Runs

| Benchmark | API calls | Est. tok/call | Total gen tokens | 7B/8B (1 GPU) | 14B/16B (1 GPU) | 32B (1-2 GPUs) | 70B (2 GPUs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ToM-Bench (x5 shuffle) | 2,860 x 5 = 14,300 | ~700 | ~10M | ~1.5 h | ~3 h | ~6 h | ~10 h |
| EQ-Bench (x10 iterations) | 171 x 10 = 1,710 | ~600 | ~1M | ~0.3 h | ~0.5 h | ~1 h | ~1.5 h |
| CharacterEval | 1,785 | ~700 | ~1.25M | ~0.4 h | ~0.7 h | ~1.2 h | ~2 h |
| PsychoBench | ~6,110 | ~300 | ~1.8M | ~0.5 h | ~0.8 h | ~1.5 h | ~2.5 h |

CharacterEval uses an additional Baichuan-based scoring model for evaluator-side scoring. This scoring overhead is included in the CharacterEval compute estimate.

## Total Compute Estimate

The table below reports **configuration-hours**, i.e., the estimated wall-clock time for one model variant on the listed serving configuration.

| Model Size | # Variants | Cat A config-h/variant | Cat B config-h/variant | Per Variant | Class Total |
|---|---:|---:|---:|---:|---:|
| 7B/8B (1 GPU) | 12 | ~1.8 | ~2.7 | ~4.5 | **~54 config-h** |
| 14B/16B (1 GPU) | 12 | ~3.0 | ~5.0 | ~8.0 | **~96 config-h** |
| 32B (1-2 GPUs) | 6 | ~5.7 | ~9.7 | ~15.4 | **~92 config-h** |
| 70B (2 GPUs, TP=2) | 7 | ~8.6 | ~16.0 | ~24.6 | **~172 config-h** |
| **Total** | **37** | | | | **~414 config-h** |

After accounting for model loading, server warmup, vLLM server startup overhead, and evaluator-side scoring overhead, we estimate:

> **Total GPU-hour estimate: ~750 RTX PRO 6000 Blackwell GPU-hours.**

## Breakdown by Benchmark

Percentages are relative to the ~414 base configuration-hour estimate before overhead.

| Benchmark | Total Runs | Est. config-h | % of Base Total |
|---|---:|---:|---:|
| ToM-Bench | 37 x 1 = 37 | ~160 | ~39% |
| HumanEval (greedy + n=20 + n=100) | 37 x 3 = 111 | ~103 | ~25% |
| PsychoBench | 37 x 1 = 37 | ~42 | ~10% |
| CharacterEval (incl. Baichuan scoring) | 37 x 1 = 37 | ~34 | ~8% |
| EQ-Bench | 37 x 1 = 37 | ~26 | ~6% |
| TruthfulQA (0-shot + 3-shot) | 37 x 2 = 74 | ~25 | ~6% |
| GSM8K | 37 x 1 = 37 | ~10 | ~2% |
| RoleEval | 37 x 1 = 37 | ~6 | ~1% |
| IFEval | 37 x 1 = 37 | ~4 | ~1% |
| C-Eval | 37 x 1 = 37 | ~4 | ~1% |

## GPU Hardware

Experiments were run on **8 x NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs**, each with **96GB GDDR7 memory**.

## Assumptions and Caveats

1. **Estimate only, not measured data.** Exact wall-clock runtime, vLLM server startup time, and total GPU-hours were not logged for every benchmark run.
2. Throughput estimates assume NVIDIA RTX PRO 6000 Blackwell 96GB GPUs with vLLM continuous batching at `gpu_memory_utilization=0.90-0.98`.
3. 70B models use 2-way tensor parallelism (TP=2); GPU-hours are counted per GPU.
4. 32B models may use one or two GPUs depending on checkpoint and serving configuration.
5. DeepSeek-V2-Lite-16B is a MoE model with approximately 2.4B active parameters and may run faster than dense models with the same nominal parameter count.
6. Quantized and sparse variants typically achieve higher throughput than FP16 variants, but this estimate does not separately model throughput for each compression method.
7. Model loading, warmup, and vLLM server startup overhead are included only approximately.
8. CharacterEval includes evaluator-side scoring with a Baichuan-based scoring model.
9. With eight GPUs available, multiple models can run in parallel using distinct ports, reducing wall-clock time relative to the summed configuration-hours.
