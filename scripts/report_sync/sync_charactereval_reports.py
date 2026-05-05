#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
CHAR_ROOT = ROOT / "paper_benchmarks" / "09_CharacterEval" / "benchmark"
REPORT_ROOT = ROOT / "report"
REPORT_SUMMARY_ROOT = ROOT / "report_summary"
if str(CHAR_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAR_ROOT))

from summary_utils import OFFICIAL_METRICS, compute_scores


METRIC_ORDER = OFFICIAL_METRICS


@dataclass(frozen=True)
class ModelConfig:
    benchmark_dir: str
    summary_filename: str
    report_dir: str
    variants: tuple[str, ...]


MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig(
        benchmark_dir="qwen2.5-7b",
        summary_filename="charactereval_results_summary.csv",
        report_dir="qwen2.5-7b",
        variants=(
            "Qwen2.5-7B-FP16",
            "Qwen2.5-7B-BNB-4bit",
            "Qwen2.5-7B-AWQ",
            "Qwen2.5-7B-GPTQ-INT4",
            "Qwen2.5-7B-2:4-Sparse",
            "Qwen2.5-7B-Unstructured-Sparse",
        ),
    ),
    ModelConfig(
        benchmark_dir="qwen2.5-14b",
        summary_filename="charactereval_results_summary.csv",
        report_dir="qwen2.5-14b",
        variants=(
            "Qwen2.5-14B-FP16",
            "Qwen2.5-14B-BNB-4bit",
            "Qwen2.5-14B-GPTQ-INT4",
            "Qwen2.5-14B-AWQ",
            "Qwen2.5-14B-Trim",
            "Qwen2.5-14B-2:4-Sparse",
            "Qwen2.5-14B-Unstructured-Sparse",
        ),
    ),
    ModelConfig(
        benchmark_dir="qwen2.5-32b",
        summary_filename="character_eval_results_summary.csv",
        report_dir="qwen2.5-32b",
        variants=(
            "Qwen2.5-32B-FP16",
            "Qwen2.5-32B-BNB-4bit",
            "Qwen2.5-32B-AWQ",
            "Qwen2.5-32B-GPTQ-INT4",
            "Qwen2.5-32B-2:4-Sparse",
            "Qwen2.5-32B-Unstructured-Sparse",
        ),
    ),
    ModelConfig(
        benchmark_dir="llama3.1-8b",
        summary_filename="charactereval_results_summary.csv",
        report_dir="llama3.1-8b",
        variants=(
            "Llama3.1-8B-FP16",
            "Llama3.1-8B-BNB-4bit",
            "Llama3.1-8B-GPTQ-INT4",
            "Llama3.1-8B-AWQ",
            "Llama3.1-8B-2:4-Sparse",
            "Llama3.1-8B-Unstructured-Sparse",
        ),
    ),
    ModelConfig(
        benchmark_dir="llama70B",
        summary_filename="charactereval_results_summary.csv",
        report_dir="llama3.1-70b",
        variants=(
            "Llama-70B-FP16",
            "Llama-70B-4bit",
            "Llama-70B-AWQ",
            "Llama-70B-GPTQ-INT4",
            "Llama-70B-GPTQ-INT8",
            "Llama70B-2:4-Sparse",
            "Llama70B-Unstructured-Sparse",
        ),
    ),
    ModelConfig(
        benchmark_dir="deepseek-v2-lite-16b",
        summary_filename="charactereval_results_summary.csv",
        report_dir="deepseek-v2-lite-16b",
        variants=(
            "DeepSeek-V2-Lite-16B-FP16",
            "DeepSeek-V2-Lite-16B-BNB-4bit",
            "DeepSeek-V2-Lite-16B-AWQ",
            "DeepSeek-V2-Lite-16B-2:4-Sparse",
            "DeepSeek-V2-Lite-16B-Unstructured-Sparse",
        ),
    ),
)
def compute_variant_row(results_root: Path, variant: str) -> list[str]:
    eval_path = results_root / variant / f"{variant}_evaluation.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")

    scores = compute_scores(eval_path)
    missing = [metric for metric in METRIC_ORDER if metric not in scores]
    if missing:
        raise ValueError(f"Missing scores for {variant}: {', '.join(missing)}")

    return [variant, f"{scores['Total']:.4f}", *[f"{scores[metric]:.4f}" for metric in METRIC_ORDER]]


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def sync_model(config: ModelConfig) -> list[list[str]]:
    header = ["Model", "Total", *METRIC_ORDER]
    results_root = CHAR_ROOT / config.benchmark_dir / "results"
    rows = [header]

    for variant in config.variants:
        rows.append(compute_variant_row(results_root, variant))

    benchmark_summary = CHAR_ROOT / config.benchmark_dir / config.summary_filename
    report_summary = REPORT_ROOT / config.report_dir / "CharacterEval" / "CharacterEval_results.csv"

    write_csv(benchmark_summary, rows)
    write_csv(report_summary, rows)
    return rows


def main() -> None:
    aggregate_rows: list[list[str]] | None = None

    for config in MODEL_CONFIGS:
        rows = sync_model(config)
        if aggregate_rows is None:
            aggregate_rows = [rows[0]]
        aggregate_rows.extend(rows[1:])

    if aggregate_rows is None:
        raise RuntimeError("No CharacterEval rows generated")

    write_csv(REPORT_SUMMARY_ROOT / "CharacterEval.csv", aggregate_rows)


if __name__ == "__main__":
    main()
