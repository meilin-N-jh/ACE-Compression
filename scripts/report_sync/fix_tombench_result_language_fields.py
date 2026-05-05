#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOMBENCH_ROOT = ROOT / "paper_benchmarks" / "07_ToM-Bench" / "benchmark"

import sys

sys.path.insert(0, str(TOMBENCH_ROOT))

from result_record_utils import localized_record_fields


ALL_TASKS = [
    line.strip().removeprefix("data/").removesuffix(".jsonl")
    for line in (TOMBENCH_ROOT / "all_tasks.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def resolve_task_name(filename: str) -> str | None:
    suffix = "_results.jsonl"
    if not filename.endswith(suffix):
        return None
    for task in sorted(ALL_TASKS, key=len, reverse=True):
        prefix = f"{task}_"
        if filename.startswith(prefix):
            return task
    return None


def load_task_rows(task_name: str) -> list[dict]:
    path = TOMBENCH_ROOT / "data" / f"{task_name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rewrite_file(path: Path) -> bool:
    task_name = resolve_task_name(path.name)
    if task_name is None:
        return False

    rows = load_task_rows(task_name)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = False

    for item in lines:
        question_idx = item.get("question_idx")
        if not isinstance(question_idx, int) or not (0 <= question_idx < len(rows)):
            continue
        localized = localized_record_fields(rows[question_idx], "en")
        for key, value in localized.items():
            if item.get(key) != value:
                item[key] = value
                changed = True

    if changed:
        with path.open("w", encoding="utf-8") as handle:
            for item in lines:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return changed


def main() -> int:
    changed_count = 0
    for path in TOMBENCH_ROOT.glob("*/results_en/*_results.jsonl"):
        if rewrite_file(path):
            changed_count += 1
            print(path)
    print(f"rewritten={changed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
