# Reproducibility Notes

This package supports verification of processed benchmark results used in ACE-Compression. It provides benchmark evaluation code, compression scripts, model serving templates, processed result files, and validation utilities. It does not provide a turnkey benchmark rerun package — users must obtain model checkpoints and benchmark datasets from their original sources.

## Reproducibility Scope

The canonical result artifacts are the processed CSV files under `results/processed/`. The validation workflow checks file presence, row/column counts, and obvious key columns for these processed files.

The package includes:

- **Benchmark evaluation code**: Launcher scripts and evaluation runners in `paper_benchmarks/` that were used to produce the paper's results
- **Compression code**: SparseGPT pruning scripts in `compression/pruning_scripts/` that were used to create compressed model variants
- **Model serving templates**: vLLM startup scripts in `serving/` documenting the exact serving configurations
- **Processed results**: Canonical CSV files in `results/processed/`
- **Report aggregation utilities**: Scripts in `scripts/report_sync/` for aggregating raw outputs into processed CSV files
- **Configuration documentation**: Model, compression, and benchmark configs in `configs/`
- **Validation script**: `scripts/prepare_results/check_processed_results.py`

## Model Checkpoints

The processed tables cover the following model families and sizes:

| Family | Sizes |
|---|---|
| Qwen2.5 | 7B, 14B, 32B |
| Llama3.1 | 8B, 70B |
| DeepSeek-V2-Lite | 16B |

Model checkpoints must be obtained from Hugging Face. See `configs/models.yaml` for source URLs, license terms, and variant summaries. Exact checkpoint revisions were not logged.

Users must place or symlink checkpoints under the expected model directory structure (documented in `WORKFLOW.md`) or update paths in launcher scripts accordingly.

## Compression Methods

The following compression methods were evaluated:

| Method | Description | Tooling |
|---|---|---|
| FP16 | Full-precision baseline | N/A |
| BNB-4bit | bitsandbytes 4-bit quantization | bitsandbytes |
| AWQ | Activation-aware quantization | AWQ / llm-awq |
| GPTQ-INT4 | GPTQ 4-bit quantization | AutoGPTQ |
| GPTQ-INT8 | GPTQ 8-bit quantization | AutoGPTQ |
| 2:4-Sparse | 2:4 structured sparsity (50%) | SparseGPT / llmcompressor |
| Unstructured-Sparse | Unstructured sparsity (50%) | SparseGPT / llmcompressor |
| Trim | Trimming variant | Model-specific |

Compression scripts are in `compression/pruning_scripts/`. Calibration data: yahma/alpaca-cleaned (CC-BY-4.0). Exact tool versions, commits, and hyperparameters were not systematically logged beyond what is documented in the scripts themselves.

## Benchmark Evaluation Workflow

### Category A: Direct lm-eval Benchmarks

IFEval, GSM8K, HumanEval, TruthfulQA, C-Eval, RoleEval are evaluated directly via lm-evaluation-harness (included in `paper_benchmarks/00_shared/`). Launcher scripts are in `paper_benchmarks/<benchmark>/<model>/`.

### Category B: vLLM Service Benchmarks

ToM-Bench, EQ-Bench, CharacterEval, PsychoBench require a running vLLM OpenAI-compatible API server. The workflow is:

1. Start vLLM server: `bash serving/<model>/start_vllm_<variant>.sh`
2. Run evaluation: `bash paper_benchmarks/<benchmark>/<model>/run_<variant>.sh`
3. Score/aggregate results

### Evaluation Configuration

Key evaluation parameters are documented in `configs/benchmarks.yaml` and the launcher scripts themselves. Notable settings:

- TruthfulQA: 0-shot and 3-shot with multiple fewshot seeds (1234, 5678, 9012)
- GSM8K: temperature=0, top_p=1, gsm8k_cot task
- HumanEval: humaneval_instruct task, 0-shot
- EQ-Bench: 10 iterations, results averaged
- ToM-Bench: seed=42, shuffle=5, ZH language
- PsychoBench: shuffle_count=1, test_count=10

## Processed Result Files

All canonical processed result files are in `results/processed/`:

| Benchmark | Processed file |
|---|---|
| IFEval | `results/processed/IFEval.csv` |
| GSM8K | `results/processed/gsm8k.csv` |
| HumanEval | `results/processed/human-eval.csv` |
| TruthfulQA | `results/processed/TruthfulQA.csv` |
| C-Eval | `results/processed/C-eval.csv` |
| RoleEval | `results/processed/role-eval.csv` |
| ToM-Bench ability | `results/processed/ToM-Bench_ability.csv` |
| ToM-Bench task | `results/processed/ToM-Bench_task.csv` |
| ToM-Bench subability | `results/processed/ToM-Bench_subability.csv` |
| EQ-Bench | `results/processed/eq-bench.csv` |
| PsychoBench | `results/processed/Psychobench.csv` |
| PsychoBench 16P | `results/processed/Psychobench_16P.csv` |
| CharacterEval | `results/processed/CharacterEval.csv` |

`CharacterEval.csv` is the canonical processed CharacterEval result file used in this package.

## Result Aggregation / Validation Workflow

Report aggregation scripts in `scripts/report_sync/` were used to produce processed CSV files from raw benchmark outputs:

| Script | Benchmarks covered |
|---|---|
| `sync_numeric_reports.py` | IFEval, C-Eval, TruthfulQA |
| `sync_gsm8k_reports.py` | GSM8K |
| `sync_charactereval_reports.py` | CharacterEval |
| `sync_eq_tom_reports.py` | EQ-Bench, ToM-Bench |
| `sync_psychobench_reports.py` | PsychoBench |

Validation of processed results:

```bash
python3 scripts/prepare_results/check_processed_results.py
```

The script checks file presence, row/column counts, and key columns without modifying data.

## Compute Resources

The processed results were generated from evaluations described in the main paper and appendix. Evaluations used NVIDIA GPUs (various models depending on checkpoint size, from single-GPU for 7B/8B models to multi-GPU for 70B). 

## What Is Not Redistributed

- Model weights and tokenizer files
- Original benchmark datasets (IFEval, GSM8K, HumanEval, TruthfulQA, C-Eval, RoleEval, ToM-Bench, EQ-Bench, CharacterEval, PsychoBench)

## Third-Party Assets and Licenses

External models, benchmarks, datasets, and software remain under their original licenses or terms. See `ASSETS.md` for source and license/terms information. The MIT license in `LICENSE` applies only to ACE-Compression-authored supplementary scripts and documentation.

Key license notes:

- RoleEval: ScienceDB dataset / CC BY-NC-SA 4.0; arXiv paper CC BY-SA 4.0
- PsychoBench: GPL-3.0 with research-use / no-commercial-use note in README
- Processed result files are included for research verification and do not relicense third-party assets
