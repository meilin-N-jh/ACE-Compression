#!/usr/bin/env python3
"""Fill EN columns in existing official-format ToM-Bench summary CSVs."""

from __future__ import annotations

import csv
from pathlib import Path

from generate_en_summary_tables import (
    ABILITY_ORDER,
    GROUPS,
    compute_task_stats,
    format_pct,
    load_records,
    load_task_metadata,
    validate_records,
)


OFFICIAL_TASKS = [
    "Faux-pas Recognition Test",
    "Unexpected Outcome Test",
    "Persuasion Story Task",
    "False Belief Task",
    "Ambiguous Story Task",
    "Strange Story Task",
    "Hinting Task Test",
    "Scalar Implicature Test",
]

ROOT = Path(__file__).resolve().parent


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        return reader.fieldnames, rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_en_results() -> dict[str, dict[str, dict[str, str]]]:
    _, task_metadata, task_to_ability, ability_to_tasks = load_task_metadata()
    all_tasks = list(task_metadata.keys())
    results: dict[str, dict[str, dict[str, str]]] = {}

    for group in GROUPS:
        group_dir = ROOT / group.directory
        results_dir = group_dir / "results_en"
        model_data: dict[str, dict[str, str]] = {}

        for variant in group.variants:
            valid_task_stats: dict[str, dict[str, float | int]] = {}
            task_cells: dict[str, str] = {}
            ability_cells: dict[str, str] = {}

            for task in all_tasks:
                result_path = results_dir / f"{task}_{variant}_results.jsonl"
                if not result_path.exists():
                    continue

                records = load_records(result_path)
                valid, _, _ = validate_records(task, records, task_metadata[task])
                if not valid:
                    continue

                stats = compute_task_stats(records)
                valid_task_stats[task] = stats
                if task in OFFICIAL_TASKS:
                    task_cells[task] = format_pct(float(stats["accuracy"]), int(stats["total"]))

            for ability in ABILITY_ORDER:
                ability_tasks = ability_to_tasks.get(ability, [])
                if ability_tasks and all(task in valid_task_stats for task in ability_tasks):
                    correct = sum(int(valid_task_stats[task]["correct"]) for task in ability_tasks)
                    total = sum(int(valid_task_stats[task]["total"]) for task in ability_tasks)
                    ability_cells[ability] = format_pct(correct / total, total)

            model_data[variant] = {
                **{f"task::{task}": value for task, value in task_cells.items()},
                **{f"ability::{ability}": value for ability, value in ability_cells.items()},
            }

        results[group.directory] = model_data

    return results


def fill_task_csv(path: Path, model_data: dict[str, dict[str, str]]) -> None:
    fieldnames, rows = read_csv(path)
    for row in rows:
        model_name = row["Model"]
        en_values = model_data.get(model_name, {})
        for task in OFFICIAL_TASKS:
            column = f"{task} (EN)"
            if column in row:
                row[column] = en_values.get(f"task::{task}", "")
    write_csv(path, fieldnames, rows)


def fill_ability_csv(path: Path, model_data: dict[str, dict[str, str]]) -> None:
    fieldnames, rows = read_csv(path)
    for row in rows:
        model_name = row["Model"]
        en_values = model_data.get(model_name, {})
        for ability in ABILITY_ORDER:
            column = f"{ability} (EN)"
            if column in row:
                row[column] = en_values.get(f"ability::{ability}", "")
    write_csv(path, fieldnames, rows)


def main() -> None:
    en_results = build_en_results()

    for group in GROUPS:
        group_dir = ROOT / group.directory
        model_data = en_results[group.directory]
        fill_task_csv(group_dir / "tombench_task_summary_with_ci.csv", model_data)
        fill_ability_csv(group_dir / "tombench_ability_summary_with_ci.csv", model_data)


if __name__ == "__main__":
    main()
