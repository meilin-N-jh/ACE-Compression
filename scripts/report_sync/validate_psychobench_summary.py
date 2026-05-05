#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY = ROOT / "report_summary" / "Psychobench.csv"
DEFAULT_QUESTIONNAIRES = ROOT / "paper_benchmarks" / "10_PsychoBench" / "benchmark" / "questionnaires.json"
RANGE_PATTERN = re.compile(r"from\s+(-?\d+)\s+to\s+(-?\d+)", re.IGNORECASE)
MEAN_PATTERN = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class CategoryLimit:
    questionnaire: str
    category: str
    question_count: int
    answer_min: int
    answer_max: int
    score_min: float
    score_max: float
    compute_mode: str


@dataclass(frozen=True)
class Violation:
    model_group: str
    model: str
    questionnaire: str
    category: str
    cell: str
    mean_value: float
    legal_min: float
    legal_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PsychoBench summary cells against official prompt ranges."
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--questionnaires", type=Path, default=DEFAULT_QUESTIONNAIRES)
    parser.add_argument("--details-out", type=Path)
    parser.add_argument("--missing-out", type=Path)
    return parser.parse_args()


def load_limits(questionnaires_path: Path) -> dict[tuple[str, str], CategoryLimit]:
    questionnaires = json.loads(questionnaires_path.read_text(encoding="utf-8"))
    limits: dict[tuple[str, str], CategoryLimit] = {}

    for questionnaire in questionnaires:
        categories = questionnaire.get("categories")
        if not isinstance(categories, list):
            continue

        range_match = RANGE_PATTERN.search(
            f'{questionnaire.get("inner_setting", "")} {questionnaire.get("prompt", "")}'
        )
        if not range_match:
            raise ValueError(
                f"Could not infer answer range for questionnaire {questionnaire['name']}"
            )

        answer_min, answer_max = (int(value) for value in range_match.groups())
        compute_mode = questionnaire["compute_mode"]

        for category in categories:
            question_count = len(category["cat_questions"])
            if compute_mode == "AVG":
                score_min = float(answer_min)
                score_max = float(answer_max)
            elif compute_mode == "SUM":
                score_min = float(answer_min * question_count)
                score_max = float(answer_max * question_count)
            else:
                raise ValueError(
                    f"Unsupported compute_mode {compute_mode} in {questionnaire['name']}"
                )

            key = (questionnaire["name"], category["cat_name"])
            limits[key] = CategoryLimit(
                questionnaire=questionnaire["name"],
                category=category["cat_name"],
                question_count=question_count,
                answer_min=answer_min,
                answer_max=answer_max,
                score_min=score_min,
                score_max=score_max,
                compute_mode=compute_mode,
            )

    return limits


def model_group(model_name: str) -> str:
    if model_name.startswith("Qwen2.5-7B"):
        return "Qwen2.5-7B"
    if model_name.startswith("Qwen2.5-14B"):
        return "Qwen2.5-14B"
    if model_name.startswith("Qwen2.5-32B"):
        return "Qwen2.5-32B"
    if model_name.startswith("Llama3.1-8B"):
        return "Llama3.1-8B"
    if model_name.startswith("Llama-70B") or model_name.startswith("Llama70B"):
        return "Llama-70B"
    if model_name.startswith("DeepSeek-V2-Lite-16B"):
        return "DeepSeek-V2-Lite-16B"
    return "UNKNOWN"


def parse_mean(cell: str) -> float | None:
    match = MEAN_PATTERN.match(cell)
    if not match:
        return None
    return float(match.group(1))


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    limits = load_limits(args.questionnaires)

    with args.summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 3:
        raise ValueError(f"Summary file is missing data rows: {args.summary}")

    header_questionnaires = rows[0]
    header_categories = rows[1]
    data_rows = rows[2:]

    violations: list[Violation] = []
    missing_cells: list[list[str]] = []
    variants_with_violations: set[str] = set()
    groups_with_violations: set[str] = set()
    questionnaire_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}

    for row in data_rows:
        model_name = row[0]
        group_name = model_group(model_name)

        for column_index in range(1, min(len(row), len(header_questionnaires))):
            key = (header_questionnaires[column_index], header_categories[column_index])
            if key not in limits:
                continue

            cell = row[column_index].strip()
            if cell == "N/A":
                missing_cells.append(
                    [group_name, model_name, key[0], key[1], "all_variants_check_pending"]
                )
                continue

            mean_value = parse_mean(cell)
            if mean_value is None:
                continue

            limit = limits[key]
            if mean_value < limit.score_min or mean_value > limit.score_max:
                violations.append(
                    Violation(
                        model_group=group_name,
                        model=model_name,
                        questionnaire=key[0],
                        category=key[1],
                        cell=cell,
                        mean_value=mean_value,
                        legal_min=limit.score_min,
                        legal_max=limit.score_max,
                    )
                )
                variants_with_violations.add(model_name)
                groups_with_violations.add(group_name)
                questionnaire_counts[key[0]] = questionnaire_counts.get(key[0], 0) + 1
                group_counts[group_name] = group_counts.get(group_name, 0) + 1

    all_na_columns: list[tuple[str, str]] = []
    for column_index in range(1, len(header_questionnaires)):
        cells = [row[column_index].strip() for row in data_rows]
        if cells and all(cell == "N/A" for cell in cells):
            all_na_columns.append((header_questionnaires[column_index], header_categories[column_index]))

    group_sizes: dict[str, int] = {}
    for row in data_rows:
        group_name = model_group(row[0])
        group_sizes[group_name] = group_sizes.get(group_name, 0) + 1

    print(f"Summary file: {args.summary}")
    print(
        f"Model variants: {len(data_rows)} across {len(group_sizes)} groups "
        f"({', '.join(f'{name}={size}' for name, size in sorted(group_sizes.items()))})"
    )
    print(
        f"Out-of-range cells: {len(violations)} across "
        f"{len(variants_with_violations)}/{len(data_rows)} variants and "
        f"{len(groups_with_violations)}/{len(group_sizes)} groups"
    )
    print(
        "By questionnaire: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(
                questionnaire_counts.items(), key=lambda item: (-item[1], item[0])
            )
        )
    )
    print(
        "By group: "
        + ", ".join(
            f"{name}={count}"
            for name, count in sorted(group_counts.items(), key=lambda item: item[0])
        )
    )
    print(
        "All-N/A columns: "
        + ", ".join(f"{questionnaire}/{category}" for questionnaire, category in all_na_columns)
    )

    if args.details_out:
        detail_rows = [[
            "model_group",
            "model",
            "questionnaire",
            "category",
            "cell",
            "mean_value",
            "legal_min",
            "legal_max",
        ]]
        for violation in violations:
            detail_rows.append([
                violation.model_group,
                violation.model,
                violation.questionnaire,
                violation.category,
                violation.cell,
                f"{violation.mean_value:g}",
                f"{violation.legal_min:g}",
                f"{violation.legal_max:g}",
            ])
        write_csv(args.details_out, detail_rows)
        print(f"Wrote detail report: {args.details_out}")

    if args.missing_out:
        missing_rows = [["questionnaire", "category", "status"]]
        for questionnaire, category in all_na_columns:
            missing_rows.append([questionnaire, category, "all_variants_N/A"])
        write_csv(args.missing_out, missing_rows)
        print(f"Wrote missing-column report: {args.missing_out}")


if __name__ == "__main__":
    main()
