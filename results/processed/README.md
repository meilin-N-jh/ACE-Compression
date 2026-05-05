# Processed Results

The files in this directory are the canonical processed benchmark result files used to verify the submitted ACE-Compression paper. They are processed evaluation outputs, not original benchmark datasets. `CharacterEval.csv` is the canonical processed CharacterEval result file used in this package.

Numeric values were not edited during packaging.

## Included Files

| Canonical file | Notes |
| --- | --- |
| `IFEval.csv` | Processed IFEval metrics. |
| `gsm8k.csv` | Processed GSM8K metrics. |
| `human-eval.csv` | Processed HumanEval pass@k metrics. |
| `TruthfulQA.csv` | Processed TruthfulQA metrics. |
| `C-eval.csv` | Processed C-Eval metrics. |
| `role-eval.csv` | Processed RoleEval metrics. |
| `ToM-Bench_ability.csv` | ToM-Bench ability-level summary. |
| `ToM-Bench_task.csv` | ToM-Bench task-level summary. |
| `ToM-Bench_subability.csv` | ToM-Bench subability-level summary. |
| `eq-bench.csv` | Processed EQ-Bench metrics. |
| `Psychobench.csv` | Processed PsychoBench questionnaire summary. |
| `Psychobench_16P.csv` | Processed PsychoBench 16P summary. |
| `CharacterEval.csv` | Canonical processed CharacterEval summary used in this package. |

## CharacterEval Canonical File

`CharacterEval.csv` is the canonical processed CharacterEval result file used in this package.

## Provenance

These processed CSV files were generated from raw benchmark evaluation outputs using the report aggregation scripts in `scripts/report_sync/`. They are included for research verification of the paper's reported results and do not constitute redistribution of original benchmark datasets. All numeric values are preserved as originally computed.

