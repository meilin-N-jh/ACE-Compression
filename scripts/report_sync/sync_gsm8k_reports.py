#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())


import csv
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(f"{ARTIFACT_ROOT}")
GSM8K_ROOT = ROOT / "paper_benchmarks" / "02_gsm8k" / "launchers"
REPORT_ROOT = ROOT / "report"
REPORT_SUMMARY_ROOT = ROOT / "report_summary"

HEADER = [
    "Model",
    "Strict Match",
    "Strict Match StdErr",
    "Flexible Match",
    "Flexible Match StdErr",
]


@dataclass(frozen=True)
class VariantConfig:
    display_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class GroupConfig:
    benchmark_dir: str
    report_dir: str
    variants: tuple[VariantConfig, ...]


GROUPS: tuple[GroupConfig, ...] = (
    GroupConfig(
        benchmark_dir="qwen2.5-7b",
        report_dir="qwen2.5-7b",
        variants=(
            VariantConfig("Qwen2.5-7B-Instruct-FP16", ("fp16",)),
            VariantConfig("Qwen2.5-7B-Instruct-BNB-4bit", ("bnb_4bit",)),
            VariantConfig("Qwen2.5-7B-Instruct-AWQ", ("awq",)),
            VariantConfig("Qwen2.5-7B-Instruct-GPTQ-INT4", ("gptq_int4",)),
            VariantConfig("Qwen2.5-7B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
            VariantConfig("Qwen2.5-7B-Instruct-Unstructured-Sparse", ("unstructured_sparse",)),
        ),
    ),
    GroupConfig(
        benchmark_dir="qwen2.5-14b",
        report_dir="qwen2.5-14b",
        variants=(
            VariantConfig("Qwen2.5-14B-Instruct-FP16", ("fp16",)),
            VariantConfig("Qwen2.5-14B-Instruct-BNB-4bit", ("bnb_4bit",)),
            VariantConfig("Qwen2.5-14B-Instruct-AWQ", ("awq",)),
            VariantConfig("Qwen2.5-14B-Instruct-GPTQ-INT4", ("gptq_int4",)),
            VariantConfig("Qwen2.5-14B-Instruct-Trim", ("trim",)),
            VariantConfig("Qwen2.5-14B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
            VariantConfig("Qwen2.5-14B-Instruct-Unstructured-Sparse", ("unstructured_sparse",)),
        ),
    ),
    GroupConfig(
        benchmark_dir="qwen2.5-32b",
        report_dir="qwen2.5-32b",
        variants=(
            VariantConfig("Qwen2.5-32B-Instruct-FP16", ("fp16",)),
            VariantConfig("Qwen2.5-32B-Instruct-BNB-4bit", ("bnb_4bit",)),
            VariantConfig("Qwen2.5-32B-Instruct-AWQ", ("awq",)),
            VariantConfig("Qwen2.5-32B-Instruct-GPTQ-INT4", ("gptq_int4",)),
            VariantConfig("Qwen2.5-32B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
            VariantConfig("Qwen2.5-32B-Instruct-Unstructured-Sparse", ("unstructured_sparse",)),
        ),
    ),
    GroupConfig(
        benchmark_dir="deepseek-v2-lite-16b",
        report_dir="deepseek-v2-lite-16b",
        variants=(
            VariantConfig("DeepSeek-V2-Lite-16B-FP16", ("fp16",)),
            VariantConfig("DeepSeek-V2-Lite-16B-BNB-4bit", ("bnb_4bit",)),
            VariantConfig("DeepSeek-V2-Lite-16B-AWQ", ("awq",)),
            VariantConfig("DeepSeek-V2-Lite-16B-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
            VariantConfig("DeepSeek-V2-Lite-16B-Unstructured-Sparse", ("unstructured_sparse",)),
        ),
    ),
    GroupConfig(
        benchmark_dir="llama3.1-8b",
        report_dir="llama3.1-8b",
        variants=(
            VariantConfig("Meta-Llama-3.1-8B-Instruct-FP16", ("fp16",)),
            VariantConfig("Meta-Llama-3.1-8B-Instruct-BNB-4bit", ("bnb_4bit", "4bit")),
            VariantConfig("Meta-Llama-3.1-8B-Instruct-AWQ", ("awq",)),
            VariantConfig("Meta-Llama-3.1-8B-Instruct-GPTQ-INT4", ("gptq_int4", "gptq")),
            VariantConfig("Meta-Llama-3.1-8B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
            VariantConfig("Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse", ("unstructured_sparse",)),
        ),
    ),
    GroupConfig(
        benchmark_dir="llama70B",
        report_dir="llama3.1-70b",
        variants=(
            VariantConfig("Llama-3.1-70B-Instruct-FP16", ("fp16",)),
            VariantConfig("Llama-3.1-70B-Instruct-4bit", ("bnb_4bit", "4bit")),
            VariantConfig("Llama-3.1-70B-Instruct-AWQ", ("awq",)),
            VariantConfig("Llama-3.1-70B-Instruct-GPTQ-INT4", ("gptq_int4", "gptq")),
            VariantConfig("Llama-3.1-70B-Instruct-GPTQ-INT8", ("gptq_int8",)),
            VariantConfig("Llama-3.1-70B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
            VariantConfig("Llama-3.1-70B-Instruct-Unstructured-Sparse", ("unstructured_sparse",)),
        ),
    ),
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def latest_match(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: (path.name, str(path)))


def path_matches_alias(path: Path, aliases: tuple[str, ...]) -> bool:
    name = path.name.lower()
    if not name.startswith("gsm8k_cot_") or not name.endswith(".json"):
        return False
    return any(alias.lower() in name for alias in aliases)


def collect_result_json(results_dir: Path, aliases: tuple[str, ...]) -> Path | None:
    matches = [path for path in results_dir.glob("*.json") if path_matches_alias(path, aliases)]
    return latest_match(matches)


def load_metrics(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    task = data["results"]["gsm8k_cot"]
    return [
        f"{float(task['exact_match,strict-match']):.4f}",
        f"{float(task['exact_match_stderr,strict-match']):.4f}",
        f"{float(task['exact_match,flexible-extract']):.4f}",
        f"{float(task['exact_match_stderr,flexible-extract']):.4f}",
    ]


def sync_group(config: GroupConfig) -> list[list[str]]:
    results_dir = GSM8K_ROOT / config.benchmark_dir / "results"
    rows = [HEADER]
    for variant in config.variants:
        result_json = collect_result_json(results_dir, variant.aliases)
        if not result_json:
            continue
        rows.append([variant.display_name, *load_metrics(result_json)])

    benchmark_path = GSM8K_ROOT / config.benchmark_dir / "gsm8k_results_summary.csv"
    report_path = REPORT_ROOT / config.report_dir / "gsm8k" / "gsm8k_results.csv"
    write_csv(benchmark_path, rows)
    write_csv(report_path, rows)
    return rows


def main() -> None:
    aggregate_rows = [HEADER]
    for config in GROUPS:
        rows = sync_group(config)
        aggregate_rows.extend(rows[1:])
    write_csv(REPORT_SUMMARY_ROOT / "gsm8k.csv", aggregate_rows)
    print(f"Updated {REPORT_SUMMARY_ROOT / 'gsm8k.csv'} with {len(aggregate_rows) - 1} rows")


if __name__ == "__main__":
    main()
