# ACE-Compression Project Workflow

This document describes the actual project workflow for evaluating compression methods on LLM benchmarks.

## Overview

The project evaluates LLMs under different compression methods across 10 benchmarks spanning four ability dimensions:

1. **Procedural reasoning**: IFEval, GSM8K, HumanEval
2. **Knowledge grounding**: TruthfulQA, C-Eval, RoleEval
3. **Social cognition**: ToM-Bench, EQ-Bench
4. **Persona fidelity**: CharacterEval, PsychoBench

## Package Structure

```
ACE-Compression/
├── paper_benchmarks/       # Benchmark evaluation code organized by benchmark
│   ├── 00_shared/          # Shared lm-evaluation-harness framework
│   ├── 01_IFEval/          # IFEval launchers
│   ├── 02_gsm8k/           # GSM8K launchers
│   ├── 03_human-eval/      # HumanEval benchmark code
│   ├── 04_TruthfulQA/      # TruthfulQA launchers
│   ├── 05_C-eval/          # C-Eval launchers
│   ├── 06_roleeval/        # RoleEval benchmark code & launchers
│   ├── 07_ToM-Bench/       # ToM-Bench benchmark code
│   ├── 08_eq-bench/        # EQ-Bench benchmark code
│   ├── 09_CharacterEval/   # CharacterEval benchmark code
│   └── 10_PsychoBench/     # PsychoBench benchmark code
├── compression/            # Compression / pruning / quantization code
│   └── pruning_scripts/    # SparseGPT-based pruning scripts
├── serving/                # vLLM model serving startup scripts
├── configs/                # Configuration files documenting setup
├── results/processed/      # Processed result CSV files (canonical)
├── scripts/                # Helper and report-sync utilities
│   ├── prepare_results/    # Validation script for processed results
│   └── report_sync/        # Report aggregation utilities
├── docs/                   # Submission notes
└── tables/                 # Reserved for table exports
```

## Benchmark Evaluation Workflow

### Benchmark Entry Points

Each numbered benchmark folder in `paper_benchmarks/` contains model-specific subdirectories with evaluation launcher scripts, along with shared benchmark code and task definitions at the benchmark root level.

### Benchmark Mapping Table

| Paper benchmark | Actual code/config entry point | Processed result file | Notes |
|---|---|---|---|
| IFEval | `paper_benchmarks/01_IFEval/<model>/run_*_vllm_ifeval.sh` | `results/processed/IFEval.csv` | Uses lm-eval vLLM backend |
| GSM8K | `paper_benchmarks/02_gsm8k/<model>/run_*_vllm_gsm8k.sh` | `results/processed/gsm8k.csv` | Uses gsm8k_cot task, temperature=0 |
| HumanEval | `paper_benchmarks/03_human-eval/<model>/run_*_vllm_humaneval.sh` | `results/processed/human-eval.csv` | humaneval_instruct task, 0-shot |
| TruthfulQA | `paper_benchmarks/04_TruthfulQA/<model>/run_*_vllm_truthfulqa.sh` | `results/processed/TruthfulQA.csv` | 0-shot and 3-shot with multiple seeds |
| C-Eval | `paper_benchmarks/05_C-eval/<model>/run_*_vllm_ceval.sh` | `results/processed/C-eval.csv` | Runs both valid and test sets |
| RoleEval | `paper_benchmarks/06_roleeval/<model>/run_roleeval_*.py` | `results/processed/role-eval.csv` | Bilingual role evaluation |
| ToM-Bench | `paper_benchmarks/07_ToM-Bench/<model>/run_*.sh` | `results/processed/ToM-Bench_ability.csv`, `results/processed/ToM-Bench_task.csv`, `results/processed/ToM-Bench_subability.csv` | Requires vLLM service; ZH language |
| EQ-Bench | `paper_benchmarks/08_eq-bench/<model>/config/config_*.cfg` | `results/processed/eq-bench.csv` | 10 iterations; requires vLLM service |
| CharacterEval | `paper_benchmarks/09_CharacterEval/<model>/run_*.sh` | `results/processed/CharacterEval.csv` | Two-stage: get responses then score with CharacterRM |
| PsychoBench | `paper_benchmarks/10_PsychoBench/<model>/run_*.sh` | `results/processed/Psychobench.csv`, `results/processed/Psychobench_16P.csv` | 14 questionnaires; requires vLLM service |

### Two Categories of Benchmarks

**Category A — Direct lm-eval (no vLLM service needed):**
IFEval, GSM8K, HumanEval, TruthfulQA, C-Eval, RoleEval

These benchmarks use lm-eval to directly load and evaluate models.

**Category B — Requires vLLM OpenAI-compatible API service:**
ToM-Bench, EQ-Bench, CharacterEval, PsychoBench

These benchmarks require starting a vLLM server first, then running evaluation via API calls.

### Typical Evaluation Flow (Category B)

1. Start vLLM service: `bash serving/<model>/start_vllm_<variant>.sh`
2. Verify service: `curl http://127.0.0.1:<PORT>/v1/models`
3. Run benchmark: `bash paper_benchmarks/<benchmark>/<model>/run_<variant>.sh`
4. Aggregate results using report sync scripts

## Compression / Pruning / Quantization Code

### Location

All compression code is in `compression/pruning_scripts/`.

### Supported Compression Methods

| Compression method | Actual code/config entry point | Tooling | Notes |
|---|---|---|---|
| FP16 | Baseline, no compression | N/A | Original model in FP16 precision |
| BnB-4bit | vLLM `--quantization bitsandbytes` flag | bitsandbytes | Applied at inference time via vLLM |
| AWQ | Pre-quantized AWQ model loading | AWQ / llm-awq | Loaded directly by vLLM |
| GPTQ-INT4 | Pre-quantized GPTQ model loading | AutoGPTQ | Loaded directly by vLLM |
| GPTQ-INT8 | Pre-quantized GPTQ model loading | AutoGPTQ | Loaded directly by vLLM |
| 2:4-Sparse | `compression/pruning_scripts/<model>/run_2_4_sparse.py` | SparseGPT via llmcompressor | One-shot pruning with alpaca-cleaned calibration |
| Unstructured-Sparse | `compression/pruning_scripts/<model>/run_unstructured_sparse.py` | SparseGPT via llmcompressor | 50% unstructured sparsity |
| Trim | Pre-processed trim model | N/A | Model-specific variant |

### Pruning Script Details

- **2:4 Structured Sparse**: `run_2_4_sparse.py` — Uses SparseGPTModifier with `mask_structure="2:4"`, 50% sparsity, 128 calibration samples from yahma/alpaca-cleaned
- **Unstructured Sparse**: `run_unstructured_sparse.py` — Uses SparseGPTModifier without structured mask, 50% sparsity
- **SliceGPT** (via TransformerCompression): Alternative pruning approach using `compression/pruning_scripts/TransformerCompression/`

### Model Coverage

Pruning scripts exist for: Qwen2.5-7B, Qwen2.5-14B, Qwen2.5-32B, Llama3.1-8B, Llama3-70B, DeepSeek-V2-Lite-16B.

## Configuration Files

Configuration documentation is in `configs/`:

- `models.yaml` — Model checkpoints, sources, licenses, and compressed variants
- `compression_methods.yaml` — Compression methods, tooling, and hyperparameters
- `benchmarks.yaml` — Benchmark sources, licenses, metric columns, and processed file mapping

## Processed Results

Canonical processed result files are in `results/processed/`. These are the final evaluation outputs used to verify the paper's reported results.

| Benchmark | Processed CSV |
|---|---|
| IFEval | `results/processed/IFEval.csv` |
| GSM8K | `results/processed/gsm8k.csv` |
| HumanEval | `results/processed/human-eval.csv` |
| TruthfulQA | `results/processed/TruthfulQA.csv` |
| C-Eval | `results/processed/C-eval.csv` |
| RoleEval | `results/processed/role-eval.csv` |
| ToM-Bench (ability) | `results/processed/ToM-Bench_ability.csv` |
| ToM-Bench (task) | `results/processed/ToM-Bench_task.csv` |
| ToM-Bench (subability) | `results/processed/ToM-Bench_subability.csv` |
| EQ-Bench | `results/processed/eq-bench.csv` |
| PsychoBench | `results/processed/Psychobench.csv` |
| PsychoBench 16P | `results/processed/Psychobench_16P.csv` |
| CharacterEval | `results/processed/CharacterEval.csv` |

`CharacterEval.csv` is the canonical processed CharacterEval result file.

### How Processed CSV Files Are Produced

Processed CSV files were generated from raw benchmark outputs using report aggregation scripts in `scripts/report_sync/`:

- `sync_numeric_reports.py` — Aggregates IFEval, C-Eval, TruthfulQA results
- `sync_gsm8k_reports.py` — Aggregates GSM8K results
- `sync_charactereval_reports.py` — Aggregates CharacterEval results
- `sync_eq_tom_reports.py` — Aggregates EQ-Bench and ToM-Bench results
- `sync_psychobench_reports.py` — Aggregates PsychoBench results

These scripts parse raw evaluation outputs and produce the processed summary CSV files. They are designed for the specific output format of each benchmark and may require adaptation for different benchmark versions.

## Result Validation

Run the validation script to verify processed result files:

```bash
python3 scripts/prepare_results/check_processed_results.py
```

This checks that all expected processed CSV files exist, prints row/column counts, and reports missing required columns without modifying data.

## External Assets Required

Users who wish to reproduce evaluations must obtain separately:

1. **Model checkpoints** from Hugging Face (Qwen2.5, Llama3.1, DeepSeek-V2-Lite)
2. **Benchmark datasets** from their original sources (see `configs/benchmarks.yaml` for URLs)
3. **vLLM** inference backend (`pip install vllm`)
4. **lm-evaluation-harness** (included in `paper_benchmarks/00_shared/`)
5. **bitsandbytes** for BnB-4bit quantization
6. **llmcompressor** for SparseGPT pruning
7. **RoleEval dataset** from ScienceDB

## Script Runnability

### Directly Runnable Scripts

- `scripts/prepare_results/check_processed_results.py` — Validates processed CSV files (no external deps beyond Python stdlib)
- `compression/pruning_scripts/<model>/run_2_4_sparse.py` — Runs SparseGPT 2:4 pruning (requires model checkpoint, llmcompressor)
- `compression/pruning_scripts/<model>/run_unstructured_sparse.py` — Runs SparseGPT unstructured pruning

### Command Templates (Require External Setup)

- `paper_benchmarks/*/<model>/run_*.sh` — Benchmark launcher scripts; require models, conda environments, and lm-eval
- `serving/<model>/start_vllm_*.sh` — vLLM startup scripts; require model checkpoints and vLLM installation

## Model Serving Ports

Port assignments for vLLM services (documented in serving scripts):

| Model | FP16 | BnB-4bit | AWQ | GPTQ-INT4 | 2:4-Sparse | Unstructured |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 8000 | 8001 | 8002 | 8003 | 8004 | 8005 |
| Qwen2.5-14B | 8100 | 8101 | 8102 | 8103 | 8104 | 8105 |
| Qwen2.5-32B | 8200 | 8201 | 8202 | 8203 | 8204 | 8205 |
| DeepSeek-V2-Lite | 8300 | 8301 | 8302 | 8303 | 8304 | 8305 |
| Llama3.1-8B | 8400 | 8401 | 8402 | 8403 | 8404 | 8405 |
| Llama3-70B | 8500 | 8501 | 8502 | 8503 | 8504 | 8505 |

## Environment

- Python 3.10+
- Conda environment for benchmark execution
- Separate conda environment for pruning (requires llmcompressor)
- Offline mode: `export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`
