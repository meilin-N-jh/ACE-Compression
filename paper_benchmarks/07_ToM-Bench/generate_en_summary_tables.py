#!/usr/bin/env python3
"""Generate validated English ToM-Bench summary CSVs for all model groups."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRY_TIMES = 5
ABILITY_ORDER = [
    "Belief",
    "Desire",
    "Emotion",
    "Intention",
    "Knowledge",
    "Non-Literal Communication",
]


@dataclass(frozen=True)
class GroupConfig:
    directory: str
    variants: tuple[str, ...]


GROUPS: tuple[GroupConfig, ...] = (
    GroupConfig(
        directory="qwen2.5-7b",
        variants=(
            "Qwen2.5-7B-FP16",
            "Qwen2.5-7B-BNB-4bit",
            "Qwen2.5-7B-GPTQ-INT4",
            "Qwen2.5-7B-AWQ",
            "Qwen2.5-7B-2:4-Sparse",
            "Qwen2.5-7B-Unstructured-Sparse",
        ),
    ),
    GroupConfig(
        directory="qwen2.5-14b",
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
    GroupConfig(
        directory="deepseek-v2-lite-16b",
        variants=(
            "DeepSeek-V2-Lite-16B-FP16",
            "DeepSeek-V2-Lite-16B-BNB-4bit",
            "DeepSeek-V2-Lite-16B-AWQ",
            "DeepSeek-V2-Lite-16B-2:4-Sparse",
            "DeepSeek-V2-Lite-16B-Unstructured-Sparse",
        ),
    ),
    GroupConfig(
        directory="qwen2.5-32b",
        variants=(
            "Qwen2.5-32B-FP16",
            "Qwen2.5-32B-BNB-4bit",
            "Qwen2.5-32B-GPTQ-INT4",
            "Qwen2.5-32B-AWQ",
            "Qwen2.5-32B-2:4-Sparse",
            "Qwen2.5-32B-Unstructured-Sparse",
        ),
    ),
    GroupConfig(
        directory="llama3.1-8b",
        variants=(
            "Llama3.1-8B-FP16",
            "Llama3.1-8B-BNB-4bit",
            "Llama3.1-8B-GPTQ-INT4",
            "Llama3.1-8B-AWQ",
            "Llama3.1-8B-2:4-Sparse",
            "Llama3.1-8B-Unstructured-Sparse",
        ),
    ),
    GroupConfig(
        directory="llama70B",
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
)


def compute_ci(accuracy: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    if accuracy <= 0.0 or accuracy >= 1.0:
        return accuracy, accuracy
    denominator = 1.0 + z**2 / total
    center = (accuracy + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(accuracy * (1 - accuracy) / total + z**2 / (4 * total**2)) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def format_pct(accuracy: float, total: int) -> str:
    lower, upper = compute_ci(accuracy, total)
    margin = (upper - lower) * 100 / 2
    return f"{accuracy * 100:.2f}% ± {margin:.2f}%"


def extract_answer(text: str) -> str:
    for marker in ("[[A]]", "[[B]]", "[[C]]", "[[D]]", "[A]", "[B]", "[C]", "[D]"):
        if marker in text:
            return marker.replace("[", "").replace("]", "")
    for char in reversed(text):
        if char in {"A", "B", "C", "D"}:
            return char
    return "A"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def load_task_metadata() -> tuple[list[str], dict[str, list[dict[str, object]]], dict[str, str], dict[str, list[str]]]:
    tasks: list[str] = []
    with (ROOT / "all_tasks.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            task_name = line.strip()
            if task_name.startswith("data/"):
                task_name = task_name[5:]
            if task_name.endswith(".jsonl"):
                task_name = task_name[:-6]
            if task_name:
                tasks.append(task_name)

    metadata: dict[str, list[dict[str, object]]] = {}
    task_to_ability: dict[str, str] = {}
    ability_to_tasks: dict[str, list[str]] = defaultdict(list)

    for task in tasks:
        entries: list[dict[str, object]] = []
        with (DATA_DIR / f"{task}.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                entries.append(
                    {
                        "story": normalize_text(record.get("STORY", "")),
                        "question": normalize_text(record.get("QUESTION", "")),
                        "choices": {
                            "A": normalize_text(record.get("OPTION-A", "")),
                            "B": normalize_text(record.get("OPTION-B", "")),
                            "C": normalize_text(record.get("OPTION-C", "")),
                            "D": normalize_text(record.get("OPTION-D", "")),
                        },
                    }
                )
                if task not in task_to_ability:
                    ability = record.get("能力\nABILITY", "Unknown")
                    category = ability.split(":")[0].strip() if ":" in ability else ability
                    task_to_ability[task] = category
                    ability_to_tasks[category].append(task)
        metadata[task] = entries

    return tasks, metadata, task_to_ability, dict(ability_to_tasks)


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_records(
    task: str,
    records: list[dict[str, object]],
    task_metadata: list[dict[str, object]],
) -> tuple[bool, list[str], int]:
    reasons: list[str] = []
    if not records:
        return False, ["empty_file"], 0

    expected_questions = len(task_metadata)
    empty_responses = 0
    seen_questions: set[int] = set()

    for record in records:
        qid = record.get("question_idx")
        if not isinstance(qid, int) or qid < 0 or qid >= expected_questions:
            reasons.append("invalid_question_idx")
            continue
        seen_questions.add(qid)

        expected = task_metadata[qid]
        if record.get("language") != "en":
            reasons.append("language_not_en")
        if normalize_text(record.get("story")) != expected["story"]:
            reasons.append("story_mismatch")
        if normalize_text(record.get("question")) != expected["question"]:
            reasons.append("question_mismatch")
        record_choices = record.get("choices") or {}
        normalized_record_choices = {
            "A": normalize_text(record_choices.get("A", "")),
            "B": normalize_text(record_choices.get("B", "")),
            "C": normalize_text(record_choices.get("C", "")),
            "D": normalize_text(record_choices.get("D", "")),
        }
        if normalized_record_choices != expected["choices"]:
            reasons.append("choices_mismatch")
        if not str(record.get("model_response", "")).strip():
            empty_responses += 1

    if seen_questions != set(range(expected_questions)):
        reasons.append("missing_questions")
    if len(records) != expected_questions * TRY_TIMES:
        reasons.append("unexpected_record_count")
    if empty_responses == len(records):
        reasons.append("all_model_response_empty")

    return len(reasons) == 0, sorted(set(reasons)), empty_responses


def compute_task_stats(records: list[dict[str, object]]) -> dict[str, float | int]:
    predictions: dict[int, list[str]] = defaultdict(list)
    answers: dict[int, str] = {}

    for record in records:
        qid = int(record["question_idx"])
        mapped_prediction = str(record.get("mapped_prediction") or "").strip()
        if not mapped_prediction:
            raw = str(record.get("model_response") or record.get("raw_prediction") or "")
            mapped_prediction = extract_answer(raw)
        predictions[qid].append(mapped_prediction)
        answers.setdefault(qid, str(record.get("answer", "")))

    total = len(predictions)
    correct = 0
    for qid, preds in predictions.items():
        if not preds:
            continue
        vote = Counter(preds).most_common(1)[0][0]
        if vote == answers.get(qid, ""):
            correct += 1

    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    tasks, task_metadata, task_to_ability, ability_to_tasks = load_task_metadata()

    status_rows: list[list[str]] = []
    status_header = [
        "Group",
        "Model",
        "Task",
        "State",
        "Reason",
        "Empty Responses",
        "Accuracy",
        "Correct",
        "Total",
    ]

    for group in GROUPS:
        group_dir = ROOT / group.directory
        results_dir = group_dir / "results_en"

        task_header = ["Model", "Status", "Valid Tasks", "Invalid Tasks", "Pending Tasks", "Overall"] + tasks
        ability_header = ["Model", "Status", "Valid Tasks", "Invalid Tasks", "Pending Tasks", "Overall"] + ABILITY_ORDER

        task_rows: list[list[str]] = []
        ability_rows: list[list[str]] = []

        for variant in group.variants:
            task_cells: dict[str, str] = {}
            task_stats: dict[str, dict[str, float | int]] = {}
            invalid_tasks: dict[str, str] = {}
            pending_tasks: list[str] = []

            for task in tasks:
                result_path = results_dir / f"{task}_{variant}_results.jsonl"
                if not result_path.exists():
                    pending_tasks.append(task)
                    status_rows.append([group.directory, variant, task, "pending", "missing_file", "0", "", "", ""])
                    continue

                records = load_records(result_path)
                valid, reasons, empty_responses = validate_records(task, records, task_metadata[task])
                if not valid:
                    invalid_tasks[task] = ";".join(reasons)
                    status_rows.append(
                        [
                            group.directory,
                            variant,
                            task,
                            "invalid",
                            ";".join(reasons),
                            str(empty_responses),
                            "",
                            "",
                            "",
                        ]
                    )
                    continue

                stats = compute_task_stats(records)
                task_stats[task] = stats
                task_cells[task] = format_pct(float(stats["accuracy"]), int(stats["total"]))
                status_rows.append(
                    [
                        group.directory,
                        variant,
                        task,
                        "valid",
                        "",
                        str(empty_responses),
                        f"{float(stats['accuracy']):.6f}",
                        str(int(stats["correct"])),
                        str(int(stats["total"])),
                    ]
                )

            valid_task_count = len(task_stats)
            invalid_task_count = len(invalid_tasks)
            pending_task_count = len(pending_tasks)

            if valid_task_count == len(tasks) and invalid_task_count == 0:
                status = "complete"
            elif invalid_task_count > 0:
                status = "invalid"
            else:
                status = "partial"

            if valid_task_count == len(tasks):
                total_correct = sum(int(stats["correct"]) for stats in task_stats.values())
                total_questions = sum(int(stats["total"]) for stats in task_stats.values())
                overall_task = format_pct(total_correct / total_questions, total_questions)
            else:
                overall_task = "N/A"

            task_row = [
                variant,
                status,
                str(valid_task_count),
                str(invalid_task_count),
                str(pending_task_count),
                overall_task,
            ]
            for task in tasks:
                if task in task_cells:
                    task_row.append(task_cells[task])
                elif task in invalid_tasks:
                    task_row.append(f"INVALID ({invalid_tasks[task]})")
                else:
                    task_row.append("PENDING")
            task_rows.append(task_row)

            ability_row = [
                variant,
                status,
                str(valid_task_count),
                str(invalid_task_count),
                str(pending_task_count),
                overall_task,
            ]
            for ability in ABILITY_ORDER:
                ability_tasks = ability_to_tasks.get(ability, [])
                if ability_tasks and all(task in task_stats for task in ability_tasks):
                    correct = sum(int(task_stats[task]["correct"]) for task in ability_tasks)
                    total = sum(int(task_stats[task]["total"]) for task in ability_tasks)
                    ability_row.append(format_pct(correct / total, total))
                else:
                    ability_row.append("N/A")
            ability_rows.append(ability_row)

        write_csv(group_dir / "tombench_task_summary_en_with_ci.csv", task_header, task_rows)
        write_csv(group_dir / "tombench_ability_summary_en_with_ci.csv", ability_header, ability_rows)

    write_csv(ROOT / "reports" / "tombench_en_validation_status.csv", status_header, status_rows)


if __name__ == "__main__":
    main()
