#!/usr/bin/env python3
"""Validate the processed benchmark result files without modifying them."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = ROOT / "results" / "processed"


EXPECTED_CSVS: dict[str, tuple[str, ...]] = {
    "IFEval.csv": ("Model", "Prompt-Strict", "Inst-Strict"),
    "gsm8k.csv": ("Model", "Strict Match", "Flexible Match"),
    "human-eval.csv": ("Model", "Pass@1 (Greedy)"),
    "TruthfulQA.csv": ("Model", "0shot-MC1", "0shot-MC2"),
    "C-eval.csv": ("Model Name", "Valid Acc", "Test Acc"),
    "role-eval.csv": ("Model", "0_en_chinese", "5_zh_global"),
    "ToM-Bench_ability.csv": ("Model", "Belief (ZH)", "Belief (EN)"),
    "ToM-Bench_task.csv": ("Model", "False Belief Task (ZH)", "False Belief Task (EN)"),
    "ToM-Bench_subability.csv": ("Model", "Belief: Content false beliefs (ZH)"),
    "eq-bench.csv": ("Model", "Benchmark Score"),
    "Psychobench.csv": ("Model", "BFI"),
    "Psychobench_16P.csv": ("Model", "Personality Type", "Count"),
    "CharacterEval.csv": ("Model", "Total", "Accuracy"),
}

def inspect_csv(path: Path, required_columns: tuple[str, ...]) -> tuple[int, int, list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return 0, 0, list(required_columns)
    header = rows[0]
    missing = [column for column in required_columns if column not in header]
    return max(len(rows) - 1, 0), len(header), missing


def main() -> int:
    print(f"Processed result directory: {PROCESSED_ROOT}")
    missing_files: list[str] = []
    column_warnings: list[str] = []

    for filename, required_columns in EXPECTED_CSVS.items():
        path = PROCESSED_ROOT / filename
        if not path.exists():
            missing_files.append(filename)
            print(f"MISSING  {filename}")
            continue
        rows, columns, missing_columns = inspect_csv(path, required_columns)
        print(f"OK       {filename}: rows={rows}, columns={columns}")
        if missing_columns:
            column_warnings.append(f"{filename}: missing columns {missing_columns}")

    if column_warnings:
        print("\nColumn warnings:")
        for warning in column_warnings:
            print(f"- {warning}")

    if missing_files:
        print("\nMissing required files:")
        for filename in missing_files:
            print(f"- {filename}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
