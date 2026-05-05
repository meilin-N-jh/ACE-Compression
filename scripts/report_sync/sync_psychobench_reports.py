#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PSYCHO_ROOT = ROOT / "paper_benchmarks" / "10_PsychoBench" / "benchmark"
REPORT_ROOT = ROOT / "report"
REPORT_SUMMARY_ROOT = ROOT / "report_summary"


QUESTIONNAIRE_ORDER: dict[str, list[str]] = {
    "BFI": ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"],
    "EPQ-R": ["Extraversion", "Pschoticism", "Neuroticism", "Lying"],
    "DTDD": ["Machiavellianism", "Psychopathy", "Narcissism"],
    "BSRI": ["Masculine", "Feminine"],
    "ECR-R": ["Attachment-related Anxiety", "Attachment-related Avoidance"],
    "EIS": ["Overall"],
    "Empathy": ["Overall"],
    "GSE": ["Overall"],
    "ICB": ["Overall"],
    "LMS": ["Factor rich", "Factor motivator", "Factor important"],
    "LOT-R": ["Overall"],
    "WLEIS": ["SEA", "OEA", "UOE", "ROE"],
    "CABIN": [
        "Mechanics/Electronics",
        "Construction/WoodWork",
        "Transportation/Machine Operation",
        "Physical/Manual Labor",
        "Protective Service",
        "Agriculture",
        "Nature/Outdoors",
        "Animal Service",
        "Athletics",
        "Engineering",
        "Physical Science",
        "Life Science",
        "Medical Science",
        "Social Science",
        "Humanities",
        "Mathematics/Statistics",
        "Information Technology",
        "Visual Arts",
        "Applied Arts and Design",
        "Performing Arts",
        "Music",
        "Writing",
        "Media",
        "Culinary Art",
        "Teaching/Education",
        "Social Service",
        "Health Care Service",
        "Religious Activities",
        "Personal Service",
        "Professional Advising",
        "Business Iniatives",
        "Sales",
        "Marketing/Advertising",
        "Finance",
        "Accounting",
        "Human Resources",
        "Office Work",
        "Management/Administration",
        "Public Speaking",
        "Politics",
        "Law",
    ],
}


QUESTIONNAIRES = tuple(QUESTIONNAIRE_ORDER.keys())
MEAN_STD_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:\\pm|±)\s*(-?\d+(?:\.\d+)?)")
PERSONALITY_PATTERN = re.compile(r"^[EI][NS][FT][JP]-[AT]$")


@dataclass(frozen=True)
class PsychoVariant:
    model_name: str
    questionnaire_dir: str
    p16_dir: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    benchmark_dir: str
    report_dir: str
    benchmark_pivot_filename: str
    variants: tuple[PsychoVariant, ...]


MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig(
        benchmark_dir="qwen2.5-7b",
        report_dir="qwen2.5-7b",
        benchmark_pivot_filename="qwen25_7b_results_pivot.csv",
        variants=(
            PsychoVariant("Qwen2.5-7B-FP16", "Qwen2.5-7B-FP16"),
            PsychoVariant("Qwen2.5-7B-BNB-4bit", "Qwen2.5-7B-BNB-4bit"),
            PsychoVariant("Qwen2.5-7B-GPTQ-INT4", "Qwen2.5-7B-GPTQ-INT4"),
            PsychoVariant("Qwen2.5-7B-AWQ", "Qwen2.5-7B-AWQ"),
            PsychoVariant("Qwen2.5-7B-2:4-Sparse", "Qwen2.5-7B-2:4-Sparse"),
            PsychoVariant("Qwen2.5-7B-Unstructured-Sparse", "Qwen2.5-7B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        benchmark_dir="qwen2.5-14b",
        report_dir="qwen2.5-14b",
        benchmark_pivot_filename="qwen25_14b_results_pivot.csv",
        variants=(
            PsychoVariant("Qwen2.5-14B-FP16", "Qwen2.5-14B-FP16"),
            PsychoVariant("Qwen2.5-14B-BNB-4bit", "Qwen2.5-14B-BNB-4bit"),
            PsychoVariant("Qwen2.5-14B-GPTQ-INT4", "Qwen2.5-14B-GPTQ-INT4"),
            PsychoVariant("Qwen2.5-14B-AWQ", "Qwen2.5-14B-AWQ"),
            PsychoVariant("Qwen2.5-14B-Trim", "Qwen2.5-14B-Trim"),
            PsychoVariant("Qwen2.5-14B-2:4-Sparse", "Qwen2.5-14B-2:4-Sparse"),
            PsychoVariant("Qwen2.5-14B-Unstructured-Sparse", "Qwen2.5-14B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        benchmark_dir="qwen2.5-32b",
        report_dir="qwen2.5-32b",
        benchmark_pivot_filename="qwen25_32b_results_pivot.csv",
        variants=(
            PsychoVariant("Qwen2.5-32B-FP16", "Qwen2.5-32B-FP16"),
            PsychoVariant("Qwen2.5-32B-BNB-4bit", "Qwen2.5-32B-BNB-4bit"),
            PsychoVariant("Qwen2.5-32B-GPTQ-INT4", "Qwen2.5-32B-GPTQ-INT4"),
            PsychoVariant("Qwen2.5-32B-AWQ", "Qwen2.5-32B-AWQ"),
            PsychoVariant("Qwen2.5-32B-2:4-Sparse", "Qwen2.5-32B-2:4-Sparse"),
            PsychoVariant("Qwen2.5-32B-Unstructured-Sparse", "Qwen2.5-32B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        benchmark_dir="llama3.1-8b",
        report_dir="llama3.1-8b",
        benchmark_pivot_filename="llama31_8b_results_pivot.csv",
        variants=(
            PsychoVariant("Llama3.1-8B-FP16", "Llama3.1-8B-FP16"),
            PsychoVariant("Llama3.1-8B-BNB-4bit", "Llama3.1-8B-BNB-4bit"),
            PsychoVariant("Llama3.1-8B-GPTQ-INT4", "Llama3.1-8B-GPTQ-INT4"),
            PsychoVariant("Llama3.1-8B-AWQ", "Llama3.1-8B-AWQ"),
            PsychoVariant("Llama3.1-8B-2:4-Sparse", "Llama3.1-8B-2:4-Sparse"),
            PsychoVariant("Llama3.1-8B-Unstructured-Sparse", "Llama3.1-8B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        benchmark_dir="llama70B",
        report_dir="llama3.1-70b",
        benchmark_pivot_filename="llama70b_results_pivot.csv",
        variants=(
            PsychoVariant("Llama-70B-FP16", "Llama-70B-FP16"),
            PsychoVariant("Llama-70B-4bit", "Llama-70B-4bit"),
            PsychoVariant("Llama-70B-AWQ", "Llama-70B-AWQ", "Llama-70B-AWQ-16P"),
            PsychoVariant("Llama-70B-GPTQ-INT4", "Llama-70B-GPTQ-INT4"),
            PsychoVariant("Llama-70B-GPTQ-INT8", "Llama-70B-GPTQ-INT8"),
            PsychoVariant("Llama70B-2:4-Sparse", "Llama70B-2:4-Sparse"),
            PsychoVariant("Llama70B-Unstructured-Sparse", "Llama70B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        benchmark_dir="deepseek-v2-lite-16b",
        report_dir="deepseek-v2-lite-16b",
        benchmark_pivot_filename="psychobench_results_pivot.csv",
        variants=(
            PsychoVariant("DeepSeek-V2-Lite-16B-FP16", "DeepSeek-V2-Lite-16B-FP16"),
            PsychoVariant("DeepSeek-V2-Lite-16B-BNB-4bit", "DeepSeek-V2-Lite-16B-BNB-4bit"),
            PsychoVariant("DeepSeek-V2-Lite-16B-AWQ", "DeepSeek-V2-Lite-16B-AWQ"),
            PsychoVariant("DeepSeek-V2-Lite-16B-2:4-Sparse", "DeepSeek-V2-Lite-16B-2:4-Sparse"),
            PsychoVariant("DeepSeek-V2-Lite-16B-Unstructured-Sparse", "DeepSeek-V2-Lite-16B-Unstructured-Sparse"),
        ),
    ),
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def format_average(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def parse_questionnaire_md(path: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.startswith("|"):
            continue
        parts = [part.strip() for part in raw_line.split("|")[1:-1]]
        if len(parts) < 2:
            continue
        category = parts[0]
        if category in {"Category", ":---:", "---"}:
            continue
        match = MEAN_STD_PATTERN.search(parts[1].replace("$", ""))
        if match:
            results[category] = f"{match.group(1)} ± {match.group(2)}"
    return results


def parse_16p_md(path: Path) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {
            "Extraverted": [],
            "Intuitive": [],
            "Thinking": [],
            "Judging": [],
            "Assertive": [],
        }
    )

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.startswith("|"):
            continue
        parts = [part.strip() for part in raw_line.split("|")[1:-1]]
        if len(parts) < 8:
            continue
        if parts[0] in {"", "Avg"} or not parts[0].isdigit():
            continue

        personality_type = parts[1]
        role = parts[2]
        if not PERSONALITY_PATTERN.fullmatch(personality_type):
            continue

        try:
            values = [float(parts[index]) for index in range(3, 8)]
        except ValueError:
            continue

        scores = grouped[(personality_type, role)]
        for key, value in zip(scores.keys(), values):
            scores[key].append(value)

    rows: list[dict[str, str]] = []
    for (personality_type, role), scores in sorted(grouped.items()):
        rows.append(
            {
                "Personality Type": personality_type,
                "Role": role,
                "Count": str(len(scores["Extraverted"])),
                "Extraverted": format_average(sum(scores["Extraverted"]) / len(scores["Extraverted"])),
                "Intuitive": format_average(sum(scores["Intuitive"]) / len(scores["Intuitive"])),
                "Thinking": format_average(sum(scores["Thinking"]) / len(scores["Thinking"])),
                "Judging": format_average(sum(scores["Judging"]) / len(scores["Judging"])),
                "Assertive": format_average(sum(scores["Assertive"]) / len(scores["Assertive"])),
            }
        )
    return rows


def build_pivot_rows(model_rows: list[tuple[str, dict[str, dict[str, str]]]]) -> list[list[str]]:
    header_row_1 = ["Model"]
    header_row_2 = [""]
    for questionnaire, dimensions in QUESTIONNAIRE_ORDER.items():
        header_row_1.extend([questionnaire] * len(dimensions))
        header_row_2.extend(dimensions)

    rows: list[list[str]] = [header_row_1, header_row_2]
    for model_name, questionnaire_results in model_rows:
        row = [model_name]
        for questionnaire, dimensions in QUESTIONNAIRE_ORDER.items():
            values = questionnaire_results.get(questionnaire, {})
            row.extend(values.get(dimension, "N/A") for dimension in dimensions)
        rows.append(row)
    return rows


def sync_model(config: ModelConfig) -> tuple[list[list[str]], list[list[str]]]:
    benchmark_dir = PSYCHO_ROOT / config.benchmark_dir
    results_root = benchmark_dir / "results"

    questionnaire_rows: list[tuple[str, dict[str, dict[str, str]]]] = []
    p16_rows: list[list[str]] = [["Model", "Personality Type", "Role", "Count", "Extraverted", "Intuitive", "Thinking", "Judging", "Assertive"]]

    for variant in config.variants:
        questionnaire_dir = results_root / variant.questionnaire_dir
        questionnaire_results: dict[str, dict[str, str]] = {}

        for questionnaire in QUESTIONNAIRES:
            md_path = questionnaire_dir / f"{variant.questionnaire_dir}-{questionnaire}.md"
            if md_path.exists():
                questionnaire_results[questionnaire] = parse_questionnaire_md(md_path)

        questionnaire_rows.append((variant.model_name, questionnaire_results))

        p16_dir_name = variant.p16_dir or variant.questionnaire_dir
        p16_path = results_root / p16_dir_name / f"{p16_dir_name}-16P.md"
        if p16_path.exists():
            for row in parse_16p_md(p16_path):
                p16_rows.append(
                    [
                        variant.model_name,
                        row["Personality Type"],
                        row["Role"],
                        row["Count"],
                        row["Extraverted"],
                        row["Intuitive"],
                        row["Thinking"],
                        row["Judging"],
                        row["Assertive"],
                    ]
                )

    pivot_rows = build_pivot_rows(questionnaire_rows)
    write_csv(benchmark_dir / config.benchmark_pivot_filename, pivot_rows)
    write_csv(benchmark_dir / "16p_results_summary.csv", p16_rows)

    report_dir = REPORT_ROOT / config.report_dir / "PsychoBench"
    write_csv(report_dir / "Psychobench_results.csv", pivot_rows)
    write_csv(report_dir / "16P_results.csv", p16_rows)
    return pivot_rows, p16_rows


def main() -> None:
    aggregate_pivot_rows: list[list[str]] | None = None
    aggregate_16p_rows = [["Model", "Personality Type", "Role", "Count", "Extraverted", "Intuitive", "Thinking", "Judging", "Assertive"]]

    for config in MODEL_CONFIGS:
        pivot_rows, p16_rows = sync_model(config)
        if aggregate_pivot_rows is None:
            aggregate_pivot_rows = [pivot_rows[0], pivot_rows[1]]
        aggregate_pivot_rows.extend(pivot_rows[2:])
        aggregate_16p_rows.extend(p16_rows[1:])

    if aggregate_pivot_rows is None:
        raise RuntimeError("No PsychoBench rows were generated")

    write_csv(REPORT_SUMMARY_ROOT / "Psychobench.csv", aggregate_pivot_rows)
    write_csv(REPORT_SUMMARY_ROOT / "Psychobench_16P.csv", aggregate_16p_rows)


if __name__ == "__main__":
    main()
