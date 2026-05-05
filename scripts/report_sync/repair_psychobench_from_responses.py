#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PSYCHO_ROOT = ROOT / "paper_benchmarks" / "10_PsychoBench" / "benchmark"
QUESTIONNAIRES_PATH = PSYCHO_ROOT / "questionnaires.json"
KNOWN_GROUPS = (
    "qwen2.5-7b",
    "qwen2.5-14b",
    "qwen2.5-32b",
    "llama3.1-8b",
    "llama70B",
    "deepseek-v2-lite-16b",
)
TEST_COLUMN_RE = re.compile(r"^shuffle(\d+)-test(\d+)$")
LINE_SCORE_RE = re.compile(r"(-?\d+)\s*%?\s*$")
COLON_PAIR_RE = re.compile(r"(?:^|\s)(?:statement|question|item)?\s*\d+\s*[:：]\s*(-?\d+)", re.IGNORECASE)
DOT_PAIR_RE = re.compile(r"(?:^|\s)\d+\s*[.)]\s*(-?\d+)")


sys.path.insert(0, str(PSYCHO_ROOT))
from utils import compute_statistics, convert_data  # noqa: E402


@dataclass(frozen=True)
class RepairOutcome:
    group: str
    variant: str
    questionnaire: str
    status: str
    detail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair PsychoBench raw CSVs from saved response logs without rerunning models."
    )
    parser.add_argument(
        "--mode",
        choices=("scan", "repair"),
        default="scan",
        help="Use 'scan' to check recoverability, 'repair' to rewrite CSV/MD files.",
    )
    parser.add_argument(
        "--questionnaires",
        default="ALL",
        help="Comma-separated questionnaire list. Default: ALL (excluding 16P).",
    )
    parser.add_argument(
        "--groups",
        default=",".join(KNOWN_GROUPS),
        help="Comma-separated PsychoBench group directories to process.",
    )
    parser.add_argument(
        "--sync-summary",
        action="store_true",
        help="After repair, regenerate report_summary/Psychobench.csv via sync_psychobench_reports.py.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "report_summary" / "Psychobench_repair_scan.csv",
        help="CSV report path for scan/repair outcomes.",
    )
    return parser.parse_args()


def load_questionnaires() -> dict[str, dict]:
    data = json.loads(QUESTIONNAIRES_PATH.read_text(encoding="utf-8"))
    return {item["name"]: item for item in data}


def answer_range(questionnaire: dict) -> tuple[int, int]:
    text = f'{questionnaire.get("inner_setting", "")} {questionnaire.get("prompt", "")}'
    match = re.search(r"from\s+(-?\d+)\s+to\s+(-?\d+)", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer answer range for {questionnaire['name']}")
    return int(match.group(1)), int(match.group(2))


def normalize_questionnaire_list(raw_value: str, known: set[str]) -> set[str]:
    if raw_value == "ALL":
        return {name for name in known if name != "16P"}
    requested = {item.strip() for item in raw_value.split(",") if item.strip()}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown questionnaires: {', '.join(unknown)}")
    return requested


def split_response_blocks(text: str) -> list[str]:
    return [chunk.strip() for chunk in text.split("====") if chunk.strip()]


def parse_scores_from_text(text: str, value_min: int, value_max: int) -> list[int]:
    candidates: list[list[int]] = []

    for pattern in (COLON_PAIR_RE, DOT_PAIR_RE):
        extracted = [int(match.group(1)) for match in pattern.finditer(text)]
        if extracted:
            candidates.append(extracted)

    line_scores: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LINE_SCORE_RE.search(line)
        if match:
            line_scores.append(int(match.group(1)))
    if line_scores:
        candidates.append(line_scores)

    valid_candidates = [
        scores for scores in candidates if all(value_min <= score <= value_max for score in scores)
    ]
    if valid_candidates:
        return max(valid_candidates, key=len)

    return []


def questionnaire_from_csv(csv_path: Path, known_names: list[str]) -> str | None:
    for questionnaire in known_names:
        if csv_path.stem.endswith(f"-{questionnaire}"):
            return questionnaire
    return None


def prompt_response_dirs(variant_dir: Path) -> tuple[Path, Path]:
    return variant_dir / "prompts", variant_dir / "responses"


def collect_variant_dirs(groups: list[str]) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    for group in groups:
        results_dir = PSYCHO_ROOT / group / "results"
        if not results_dir.exists():
            continue
        for variant_dir in sorted(results_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            if "_backup_" in variant_dir.name:
                continue
            if not list(variant_dir.glob("*.csv")):
                continue
            results.append((group, variant_dir))
    return results


def grouped_test_columns(fieldnames: list[str]) -> dict[int, list[str]]:
    grouped: dict[int, list[tuple[int, str]]] = {}
    for name in fieldnames:
        match = TEST_COLUMN_RE.fullmatch(name)
        if not match:
            continue
        shuffle_index = int(match.group(1))
        test_index = int(match.group(2))
        grouped.setdefault(shuffle_index, []).append((test_index, name))
    return {
        shuffle_index: [name for _, name in sorted(items)]
        for shuffle_index, items in grouped.items()
    }


def repair_questionnaire_csv(
    group: str,
    variant_dir: Path,
    csv_path: Path,
    questionnaire_name: str,
    questionnaire: dict,
    do_write: bool,
) -> RepairOutcome:
    response_dir = variant_dir / "responses"
    if not response_dir.exists():
        return RepairOutcome(group, variant_dir.name, questionnaire_name, "needs_rerun", "missing responses dir")

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames or not rows:
        return RepairOutcome(group, variant_dir.name, questionnaire_name, "needs_rerun", "empty csv")

    row_count = len(rows)
    chunk_count = math.ceil(row_count / 30)
    value_min, value_max = answer_range(questionnaire)
    test_columns = grouped_test_columns(fieldnames)
    if not test_columns:
        return RepairOutcome(group, variant_dir.name, questionnaire_name, "needs_rerun", "no test columns")

    parsed_columns: dict[str, list[int]] = {}
    notes: list[str] = []

    for shuffle_index, columns in sorted(test_columns.items()):
        response_path = response_dir / f"{variant_dir.name}-{questionnaire_name}-shuffle{shuffle_index}.txt"
        if not response_path.exists():
            return RepairOutcome(
                group,
                variant_dir.name,
                questionnaire_name,
                "needs_rerun",
                f"missing {response_path.name}",
            )

        blocks = split_response_blocks(response_path.read_text(encoding="utf-8", errors="ignore"))
        expected_blocks = len(columns) * chunk_count
        if len(blocks) < expected_blocks:
            return RepairOutcome(
                group,
                variant_dir.name,
                questionnaire_name,
                "needs_rerun",
                f"{response_path.name}: expected {expected_blocks} blocks, found {len(blocks)}",
            )
        if len(blocks) > expected_blocks:
            notes.append(f"{response_path.name}: using last {expected_blocks}/{len(blocks)} blocks")
            blocks = blocks[-expected_blocks:]

        for test_index, column_name in enumerate(columns):
            start = test_index * chunk_count
            end = start + chunk_count
            scores = parse_scores_from_text("\n".join(blocks[start:end]), value_min, value_max)
            if len(scores) != row_count:
                return RepairOutcome(
                    group,
                    variant_dir.name,
                    questionnaire_name,
                    "needs_rerun",
                    f"{response_path.name}:{column_name} parsed {len(scores)} scores, expected {row_count}",
                )
            parsed_columns[column_name] = scores

    if do_write:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(csv_path, backup_path)

        for row_index, row in enumerate(rows):
            for column_name, scores in parsed_columns.items():
                row[column_name] = str(scores[row_index])

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    detail = "; ".join(notes)
    return RepairOutcome(group, variant_dir.name, questionnaire_name, "repaired" if do_write else "recoverable", detail)


def write_minimal_md(questionnaire: dict, csv_path: Path, md_path: Path, model_label: str) -> None:
    test_data = convert_data(questionnaire, str(csv_path))
    stats = compute_statistics(questionnaire, test_data)
    crowd_columns = questionnaire["categories"][0]["crowd"]

    lines = [
        f"# {questionnaire['name']} Results",
        "",
        "| Category | "
        + f"{model_label} (n = {len(test_data)}) | "
        + " | ".join(f"{crowd['crowd_name']} (n = {crowd['n']})" for crowd in crowd_columns)
        + " |",
        "| :---: | " + " | ".join(":---:" for _ in range(len(crowd_columns) + 1)) + " |",
    ]

    for index, category in enumerate(questionnaire["categories"]):
        mean_value, std_value, _ = stats[index]
        crowd_values = " | ".join(
            f"{crowd['mean']:.1f} $\\pm$ {crowd['std']:.1f}" for crowd in category["crowd"]
        )
        lines.append(
            f"| {category['cat_name']} | {mean_value:.1f} $\\pm$ {std_value:.1f} | {crowd_values} | "
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def backup_once(path: Path) -> None:
    if not path.exists():
        return
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def restore_from_backup(path: Path) -> bool:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        return False
    shutil.copy2(backup_path, path)
    return True


def write_placeholder_md(questionnaire_name: str, md_path: Path, detail: str) -> None:
    lines = [
        f"# {questionnaire_name} Results",
        "",
        "Repair incomplete for this questionnaire.",
        "",
        f"Reason: {detail}",
        "",
        "This entry is intentionally left without summary statistics so downstream tables render as N/A.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def regenerate_md(
    group: str,
    variant_dir: Path,
    questionnaires: dict[str, dict],
    repaired_names: set[str],
) -> list[RepairOutcome]:
    outcomes: list[RepairOutcome] = []
    for questionnaire_name in sorted(repaired_names):
        csv_path = variant_dir / f"{variant_dir.name}-{questionnaire_name}.csv"
        md_path = variant_dir / f"{variant_dir.name}-{questionnaire_name}.md"
        try:
            backup_once(md_path)
            write_minimal_md(questionnaires[questionnaire_name], csv_path, md_path, variant_dir.name)
            outcomes.append(RepairOutcome(group, variant_dir.name, questionnaire_name, "md_regenerated"))
        except Exception as exc:
            outcomes.append(RepairOutcome(group, variant_dir.name, questionnaire_name, "md_failed", str(exc)))
    return outcomes


def write_report(path: Path, outcomes: list[RepairOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "variant", "questionnaire", "status", "detail"])
        for outcome in outcomes:
            writer.writerow([outcome.group, outcome.variant, outcome.questionnaire, outcome.status, outcome.detail])


def main() -> None:
    args = parse_args()
    questionnaires = load_questionnaires()
    selected = normalize_questionnaire_list(args.questionnaires, set(questionnaires))
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]

    outcomes: list[RepairOutcome] = []
    variants = collect_variant_dirs(groups)
    known_names = sorted(questionnaires.keys(), key=len, reverse=True)

    for group, variant_dir in variants:
        repaired_names: set[str] = set()
        unresolved: list[RepairOutcome] = []
        for csv_path in sorted(variant_dir.glob("*.csv")):
            questionnaire_name = questionnaire_from_csv(csv_path, known_names)
            if questionnaire_name is None or questionnaire_name not in selected or questionnaire_name == "16P":
                continue
            outcome = repair_questionnaire_csv(
                group=group,
                variant_dir=variant_dir,
                csv_path=csv_path,
                questionnaire_name=questionnaire_name,
                questionnaire=questionnaires[questionnaire_name],
                do_write=args.mode == "repair",
            )
            outcomes.append(outcome)
            if args.mode == "repair":
                if outcome.status == "repaired":
                    repaired_names.add(questionnaire_name)
                elif outcome.status == "needs_rerun":
                    csv_path = variant_dir / f"{variant_dir.name}-{questionnaire_name}.csv"
                    restore_from_backup(csv_path)
                    unresolved.append(outcome)

        if args.mode == "repair":
            outcomes.extend(regenerate_md(group, variant_dir, questionnaires, repaired_names))
            for outcome in unresolved:
                md_path = variant_dir / f"{variant_dir.name}-{outcome.questionnaire}.md"
                backup_once(md_path)
                write_placeholder_md(outcome.questionnaire, md_path, outcome.detail)
                outcomes.append(
                    RepairOutcome(
                        group,
                        variant_dir.name,
                        outcome.questionnaire,
                        "md_placeholder",
                        outcome.detail,
                    )
                )

    write_report(args.report_out, outcomes)

    status_counts = Counter(outcome.status for outcome in outcomes)
    print(f"Processed variants: {len(variants)}")
    print("Outcome counts: " + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items())))
    print(f"Report: {args.report_out}")

    if args.mode == "repair" and args.sync_summary:
        sync_script = ROOT / "scripts" / "sync_psychobench_reports.py"
        namespace: dict[str, object] = {"__file__": str(sync_script), "__name__": "__repair_sync__"}
        exec(sync_script.read_text(encoding="utf-8"), namespace)
        namespace["main"]()
        print("Regenerated PsychoBench report_summary files.")


if __name__ == "__main__":
    main()
