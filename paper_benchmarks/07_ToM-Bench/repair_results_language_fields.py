#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TASKS = [
    line.strip().removeprefix("data/").removesuffix(".jsonl")
    for line in (ROOT / "all_tasks.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def match_task(filename: str) -> str | None:
    for task in TASKS:
        if filename.startswith(f"{task}_") and filename.endswith("_results.jsonl"):
            return task
    return None


def localized_fields(record: dict, language: str) -> tuple[str, str, dict[str, str]]:
    if language == "en":
        return (
            record.get("STORY", ""),
            record.get("QUESTION", ""),
            {
                "A": record.get("OPTION-A", ""),
                "B": record.get("OPTION-B", ""),
                "C": record.get("OPTION-C", ""),
                "D": record.get("OPTION-D", ""),
            },
        )
    return (
        record.get("故事", ""),
        record.get("问题", ""),
        {
            "A": record.get("选项A", ""),
            "B": record.get("选项B", ""),
            "C": record.get("选项C", ""),
            "D": record.get("选项D", ""),
        },
    )


def rewrite_file(path: Path) -> bool:
    task = match_task(path.name)
    if task is None:
        return False

    data = [
        json.loads(line)
        for line in (DATA_DIR / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    language = "en" if path.parent.name == "results_en" else "zh"

    updated = False
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question_idx = row.get("question_idx")
        if isinstance(question_idx, int) and 0 <= question_idx < len(data):
            story, question, choices = localized_fields(data[question_idx], language)
            if (
                row.get("language") != language
                or row.get("story") != story
                or row.get("question") != question
                or row.get("choices") != choices
            ):
                row["language"] = language
                row["story"] = story
                row["question"] = question
                row["choices"] = choices
                updated = True
        rows.append(row)

    if updated:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return updated


def main() -> int:
    changed = 0
    for results_dir in ROOT.glob("*/results*"):
        if not results_dir.is_dir():
            continue
        if results_dir.name not in {"results", "results_en"}:
            continue
        for path in results_dir.glob("*_results.jsonl"):
            if rewrite_file(path):
                changed += 1
                print(path)
    print(f"updated_files={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
