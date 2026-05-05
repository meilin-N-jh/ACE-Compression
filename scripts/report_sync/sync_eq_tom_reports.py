#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EQ_ROOT = ROOT / "paper_benchmarks" / "08_eq-bench" / "benchmark"
TOM_ROOT = ROOT / "paper_benchmarks" / "07_ToM-Bench" / "benchmark"
REPORT_ROOT = ROOT / "report"
REPORT_SUMMARY_ROOT = ROOT / "report_summary"


ABILITY_ORDER = [
    "Belief",
    "Desire",
    "Emotion",
    "Intention",
    "Knowledge",
    "Non-Literal Communication",
]

SUBABILITY_ORDER = [
    "Belief: Content false beliefs",
    "Belief: Location false beliefs",
    "Belief: Identity false beliefs",
    "Belief: Second-order beliefs",
    "Belief: Beliefs based action/emotions",
    "Belief: Sequence false beliefs",
    "Desire: Discrepant desires",
    "Desire: Multiple desires",
    "Desire: Desires influence on emotions and actions",
    "Desire: Desire-action contradiction",
    "Emotion: Typical emotional reactions",
    "Emotion: Atypical emotional reactions",
    "Emotion: Discrepant emotions",
    "Emotion: Mixed emotions",
    "Emotion: Hidden emotions",
    "Emotion: Moral emotions",
    "Emotion: Emotion regulation",
    "Intention: Completion of failed actions",
    "Intention: Discrepant intentions",
    "Intention: Prediction of actions",
    "Intention: Intentions explanations",
    "Knowledge: Knowledge-pretend play links",
    "Knowledge: Percepts-knowledge links",
    "Knowledge: Information-knowledge links",
    "Knowledge: Knowledge-attention links",
    "Non-Literal Communication: Irony/Sarcasm",
    "Non-Literal Communication: Egocentric lies",
    "Non-Literal Communication: White lies",
    "Non-Literal Communication: Involuntary lies",
    "Non-Literal Communication: Humor",
    "Non-Literal Communication: Faux pas",
]

OFFICIAL_TASK_ORDER = [
    "Faux-pas Recognition Test",
    "Unexpected Outcome Test",
    "Persuasion Story Task",
    "False Belief Task",
    "Ambiguous Story Task",
    "Strange Story Task",
    "Hinting Task Test",
    "Scalar Implicature Test",
]

ABILITY_PREFIX_PATTERN = re.compile(
    r"(Belief|Desire|Emotion|Intention|Knowledge|Non-Literal Communication|Non-literal communication): "
)

SUBABILITY_MERGE_MAP = {
    "Desire: Desires influence on actions": "Desire: Desires influence on emotions and actions",
    "Desire: Desires influence on emotions (beliefs)": "Desire: Desires influence on emotions and actions",
}


@dataclass(frozen=True)
class EqVariant:
    filename: str
    model_name: str


@dataclass(frozen=True)
class TomVariant:
    result_group: str
    model_name: str


@dataclass(frozen=True)
class ModelConfig:
    report_dir: str
    eq_folder: str
    tom_folder: str
    eq_variants: tuple[EqVariant, ...]
    tom_variants: tuple[TomVariant, ...]


MODEL_CONFIGS: tuple[ModelConfig, ...] = (
    ModelConfig(
        report_dir="qwen2.5-7b",
        eq_folder="qwen2.5-7b",
        tom_folder="qwen2.5-7b",
        eq_variants=(
            EqVariant("benchmark_results_qwen25_7b_fp16.csv", "Qwen2.5-7B-Instruct-FP16"),
            EqVariant("benchmark_results_qwen25_7b_bnb_4bit.csv", "Qwen2.5-7B-Instruct-BNB-4bit"),
            EqVariant("benchmark_results_qwen25_7b_awq.csv", "Qwen2.5-7B-Instruct-AWQ"),
            EqVariant("benchmark_results_qwen25_7b_gptq_int4.csv", "Qwen2.5-7B-Instruct-GPTQ-INT4"),
            EqVariant("benchmark_results_qwen25_7b_2_4_sparse.csv", "Qwen2.5-7B-Instruct-2:4-Sparse"),
            EqVariant("benchmark_results_qwen25_7b_unstructured_sparse.csv", "Qwen2.5-7B-Instruct-Unstructured-Sparse"),
        ),
        tom_variants=(
            TomVariant("Qwen2.5-7B-FP16", "Qwen2.5-7B-FP16"),
            TomVariant("Qwen2.5-7B-BNB-4bit", "Qwen2.5-7B-BNB-4bit"),
            TomVariant("Qwen2.5-7B-GPTQ-INT4", "Qwen2.5-7B-GPTQ-INT4"),
            TomVariant("Qwen2.5-7B-AWQ", "Qwen2.5-7B-AWQ"),
            TomVariant("Qwen2.5-7B-2:4-Sparse", "Qwen2.5-7B-2:4-Sparse"),
            TomVariant("Qwen2.5-7B-Unstructured-Sparse", "Qwen2.5-7B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        report_dir="qwen2.5-14b",
        eq_folder="qwen2.5-14b",
        tom_folder="qwen2.5-14b",
        eq_variants=(
            EqVariant("benchmark_results_qwen25_14b_fp16.csv", "Qwen2.5-14B-Instruct-FP16"),
            EqVariant("benchmark_results_qwen25_14b_bnb_4bit.csv", "Qwen2.5-14B-Instruct-BNB-4bit"),
            EqVariant("benchmark_results_qwen25_14b_awq.csv", "Qwen2.5-14B-Instruct-AWQ"),
            EqVariant("benchmark_results_qwen25_14b_gptq_int4.csv", "Qwen2.5-14B-Instruct-GPTQ-INT4"),
            EqVariant("benchmark_results_qwen25_14b_trim.csv", "Qwen2.5-14B-Instruct-Trim"),
            EqVariant("benchmark_results_qwen25_14b_2_4_sparse.csv", "Qwen2.5-14B-Instruct-2:4-Sparse"),
            EqVariant("benchmark_results_qwen25_14b_unstructured_sparse.csv", "Qwen2.5-14B-Instruct-Unstructured-Sparse"),
        ),
        tom_variants=(
            TomVariant("Qwen2.5-14B-FP16", "Qwen2.5-14B-FP16"),
            TomVariant("Qwen2.5-14B-BNB-4bit", "Qwen2.5-14B-BNB-4bit"),
            TomVariant("Qwen2.5-14B-GPTQ-INT4", "Qwen2.5-14B-GPTQ-INT4"),
            TomVariant("Qwen2.5-14B-AWQ", "Qwen2.5-14B-AWQ"),
            TomVariant("Qwen2.5-14B-Trim", "Qwen2.5-14B-Trim"),
            TomVariant("Qwen2.5-14B-2:4-Sparse", "Qwen2.5-14B-2:4-Sparse"),
            TomVariant("Qwen2.5-14B-Unstructured-Sparse", "Qwen2.5-14B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        report_dir="qwen2.5-32b",
        eq_folder="qwen2.5-32b",
        tom_folder="qwen2.5-32b",
        eq_variants=(
            EqVariant("benchmark_results_qwen25_32b_fp16.csv", "Qwen2.5-32B-Instruct-FP16"),
            EqVariant("benchmark_results_qwen25_32b_bnb_4bit.csv", "Qwen2.5-32B-Instruct-BNB-4bit"),
            EqVariant("benchmark_results_qwen25_32b_awq.csv", "Qwen2.5-32B-Instruct-AWQ"),
            EqVariant("benchmark_results_qwen25_32b_gptq_int4.csv", "Qwen2.5-32B-Instruct-GPTQ-INT4"),
            EqVariant("benchmark_results_qwen25_32b_2_4_sparse.csv", "Qwen2.5-32B-Instruct-2:4-Sparse"),
            EqVariant("benchmark_results_qwen25_32b_unstructured_sparse.csv", "Qwen2.5-32B-Instruct-Unstructured-Sparse"),
        ),
        tom_variants=(
            TomVariant("Qwen2.5-32B-FP16", "Qwen2.5-32B-FP16"),
            TomVariant("Qwen2.5-32B-BNB-4bit", "Qwen2.5-32B-BNB-4bit"),
            TomVariant("Qwen2.5-32B-GPTQ-INT4", "Qwen2.5-32B-GPTQ-INT4"),
            TomVariant("Qwen2.5-32B-AWQ", "Qwen2.5-32B-AWQ"),
            TomVariant("Qwen2.5-32B-2:4-Sparse", "Qwen2.5-32B-2:4-Sparse"),
            TomVariant("Qwen2.5-32B-Unstructured-Sparse", "Qwen2.5-32B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        report_dir="llama3.1-8b",
        eq_folder="llama3.1-8b",
        tom_folder="llama3.1-8b",
        eq_variants=(
            EqVariant("benchmark_results_llama3_1_8b_fp16.csv", "Meta-Llama-3.1-8B-Instruct-FP16"),
            EqVariant("benchmark_results_llama3_1_8b_4bit.csv", "Meta-Llama-3.1-8B-Instruct-BNB-4bit"),
            EqVariant("benchmark_results_llama3_1_8b_awq.csv", "Meta-Llama-3.1-8B-Instruct-AWQ"),
            EqVariant("benchmark_results_llama3_1_8b_gptq.csv", "Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"),
            EqVariant("benchmark_results_llama31_8b_2_4_sparse.csv", "Meta-Llama-3.1-8B-Instruct-2:4-Sparse"),
            EqVariant("benchmark_results_llama31_8b_unstructured_sparse.csv", "Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse"),
        ),
        tom_variants=(
            TomVariant("Llama3.1-8B-FP16", "Llama3.1-8B-FP16"),
            TomVariant("Llama3.1-8B-BNB-4bit", "Llama3.1-8B-BNB-4bit"),
            TomVariant("Llama3.1-8B-GPTQ-INT4", "Llama3.1-8B-GPTQ-INT4"),
            TomVariant("Llama3.1-8B-AWQ", "Llama3.1-8B-AWQ"),
            TomVariant("Llama3.1-8B-2:4-Sparse", "Llama3.1-8B-2:4-Sparse"),
            TomVariant("Llama3.1-8B-Unstructured-Sparse", "Llama3.1-8B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        report_dir="deepseek-v2-lite-16b",
        eq_folder="deepseek-v2-lite-16b",
        tom_folder="deepseek-v2-lite-16b",
        eq_variants=(
            EqVariant("benchmark_results_deepseek_v2_lite_fp16.csv", "DeepSeek-V2-Lite-16B-FP16"),
            EqVariant("benchmark_results_deepseek_v2_lite_bnb_4bit.csv", "DeepSeek-V2-Lite-16B-BNB-4bit"),
            EqVariant("benchmark_results_deepseek_v2_lite_awq.csv", "DeepSeek-V2-Lite-16B-AWQ"),
            EqVariant("benchmark_results_deepseek_v2_lite_2_4_sparse.csv", "DeepSeek-V2-Lite-16B-2:4-Sparse"),
            EqVariant("benchmark_results_deepseek_v2_lite_unstructured_sparse.csv", "DeepSeek-V2-Lite-16B-Unstructured-Sparse"),
        ),
        tom_variants=(
            TomVariant("DeepSeek-V2-Lite-16B-FP16", "DeepSeek-V2-Lite-16B-FP16"),
            TomVariant("DeepSeek-V2-Lite-16B-BNB-4bit", "DeepSeek-V2-Lite-16B-BNB-4bit"),
            TomVariant("DeepSeek-V2-Lite-16B-AWQ", "DeepSeek-V2-Lite-16B-AWQ"),
            TomVariant("DeepSeek-V2-Lite-16B-2:4-Sparse", "DeepSeek-V2-Lite-16B-2:4-Sparse"),
            TomVariant("DeepSeek-V2-Lite-16B-Unstructured-Sparse", "DeepSeek-V2-Lite-16B-Unstructured-Sparse"),
        ),
    ),
    ModelConfig(
        report_dir="llama3.1-70b",
        eq_folder="llama70B",
        tom_folder="llama70B",
        eq_variants=(
            EqVariant("benchmark_results_llama70b_fp16.csv", "Llama-70B-FP16"),
            EqVariant("benchmark_results_llama70b_4bit.csv", "Llama-70B-4bit"),
            EqVariant("benchmark_results_llama70b_awq.csv", "Llama-70B-AWQ"),
            EqVariant("benchmark_results_llama70b_gptq_int4.csv", "Llama-70B-GPTQ-INT4"),
            EqVariant("benchmark_results_llama70b_gptq_int8.csv", "Llama-70B-GPTQ-INT8"),
            EqVariant("benchmark_results_llama70b_2_4_sparse.csv", "Llama-70B-2:4-Sparse"),
            EqVariant("benchmark_results_llama70b_unstructured_sparse.csv", "Llama-70B-Unstructured-Sparse"),
        ),
        tom_variants=(
            TomVariant("Llama-70B-FP16", "Llama-70B-FP16"),
            TomVariant("Llama-70B-4bit", "Llama-70B-4bit"),
            TomVariant("Llama-70B-AWQ", "Llama-70B-AWQ"),
            TomVariant("Llama-70B-GPTQ-INT4", "Llama-70B-GPTQ-INT4"),
            TomVariant("Llama-70B-GPTQ-INT8", "Llama-70B-GPTQ-INT8"),
            TomVariant("Llama70B-2:4-Sparse", "Llama70B-2:4-Sparse"),
            TomVariant("Llama70B-Unstructured-Sparse", "Llama70B-Unstructured-Sparse"),
        ),
    ),
)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def parse_completed_at(value: str) -> datetime:
    return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")


def is_success_score(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def extract_failed_parseable(row: dict[str, str]) -> str:
    error = row.get("Error", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?) questions were parseable", error)
    if match:
        return match.group(1)
    return row.get("Num Questions Parseable", "FAILED")


def format_score(raw_value: str) -> str:
    try:
        return f"{float(raw_value):.2f}"
    except ValueError:
        return raw_value


def pick_eq_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    successful = [row for row in rows if is_success_score(row.get("Benchmark Score", ""))]
    candidates = successful if successful else rows
    return max(candidates, key=lambda row: parse_completed_at(row["Benchmark Completed"]))


def parse_eq_log(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    completed = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    scores = re.findall(r"Score \(v2\):\s*([^\n]+)", text)
    parseables = re.findall(r"Parseable:\s*([^\n]+)", text)
    if not completed or not scores or not parseables:
        return None
    return [
        "",  # model name will be filled by caller
        format_score(scores[-1].strip()),
        parseables[-1].strip(),
        completed[-1].split(" ")[0],
    ]


def eq_log_keyword(model_name: str) -> str:
    suffix = model_name.split("-")[-1]
    if model_name.endswith("BNB-4bit"):
        return "bnb_4bit"
    if model_name.endswith("GPTQ-INT4"):
        return "gptq_int4"
    if model_name.endswith("GPTQ-INT8"):
        return "gptq_int8"
    if model_name.endswith("2:4-Sparse"):
        return "2_4_sparse"
    if model_name.endswith("Unstructured-Sparse"):
        return "unstructured_sparse"
    if model_name.endswith("FP16"):
        return "fp16"
    if model_name.endswith("AWQ"):
        return "awq"
    if model_name.endswith("Trim"):
        return "trim"
    if model_name.endswith("4bit"):
        return "4bit"
    return suffix.lower().replace(":", "_")


def read_existing_eq_summary(path: Path) -> dict[str, dict[str, str]]:
    return {row["Model"]: row for row in load_csv_rows(path)}


def build_eq_rows(config: ModelConfig) -> tuple[list[list[str]], list[str]]:
    summary_path = EQ_ROOT / config.eq_folder / "eqbench_results_summary.csv"
    existing_rows = read_existing_eq_summary(summary_path)
    result_dir = EQ_ROOT / config.eq_folder / "results"
    folder_dir = EQ_ROOT / config.eq_folder
    output_rows: list[list[str]] = []
    warnings: list[str] = []

    for variant in config.eq_variants:
        result_path = result_dir / variant.filename
        if not result_path.exists():
            alt_path = folder_dir / variant.filename
            if alt_path.exists():
                result_path = alt_path
        selected: list[str] | None = None

        if result_path.exists():
            chosen_row = pick_eq_row(load_csv_rows(result_path))
            if chosen_row:
                score = chosen_row.get("Benchmark Score", "FAILED")
                parseable = chosen_row.get("Num Questions Parseable", "FAILED")
                if score == "FAILED":
                    parseable = extract_failed_parseable(chosen_row)
                run_date = chosen_row.get("Benchmark Completed", "").split(" ")[0]
                selected = [
                    variant.model_name,
                    format_score(score),
                    parseable,
                    run_date,
                ]
        if selected is None:
            log_key = eq_log_keyword(variant.model_name)
            log_candidates = sorted(folder_dir.glob(f"logs/eqbench_{log_key}_*.log"))
            if log_candidates:
                parsed = parse_eq_log(log_candidates[-1])
                if parsed:
                    parsed[0] = variant.model_name
                    selected = parsed
                    warnings.append(
                        f"[eq-bench] {config.eq_folder} 的 {variant.model_name} 使用 log 回填结果"
                    )
        if selected is None and variant.model_name in existing_rows:
            old = existing_rows[variant.model_name]
            selected = [
                old["Model"],
                old["Benchmark Score"],
                old["Num Questions Parseable"],
                old["Run Date"],
            ]
            warnings.append(
                f"[eq-bench] {config.eq_folder} 缺少 {variant.filename}，保留现有 summary 行 {variant.model_name}"
            )

        if selected:
            output_rows.append(selected)

    existing_names = set(existing_rows)
    generated_names = {row[0] for row in output_rows}
    missing_from_results = sorted(existing_names - generated_names)
    if missing_from_results:
        warnings.append(
            f"[eq-bench] {config.eq_folder} 这些 summary 模型未能重建: {', '.join(missing_from_results)}"
        )

    return output_rows, warnings


def compute_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0 or p in (0.0, 1.0):
        return 0.0, 0.0
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def format_tom_percent(acc: float, total: int) -> str:
    lower, upper = compute_ci(acc, total)
    margin = (upper - lower) * 100 / 2
    return f"{acc * 100:.2f}% ± {margin:.2f}%"


def format_tom_decimal(acc: float, total: int) -> str:
    lower, upper = compute_ci(acc, total)
    margin = (upper - lower) / 2
    return f"{acc:.4f} ± {margin:.4f}"


def bilingual_header(columns: list[str]) -> list[str]:
    return ["Model"] + [f"{column} (ZH)" for column in columns] + [f"{column} (EN)" for column in columns]


def bilingual_row(model_name: str, zh_values: list[str], en_values: list[str] | None = None) -> list[str]:
    if en_values is None:
        en_values = [""] * len(zh_values)
    return [model_name] + zh_values + en_values


def load_task_to_ability() -> dict[str, str]:
    task_to_ability: dict[str, str] = {}
    data_dir = TOM_ROOT / "data"
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline().strip()
        if not first_line:
            continue
        record = json.loads(first_line)
        full_ability = record.get("能力\nABILITY", "")
        ability = full_ability.split(":", 1)[0].strip() if ":" in full_ability else full_ability.strip()
        task_to_ability[path.stem] = ability
    return task_to_ability


def split_subabilities(full_ability: str) -> list[str]:
    full_ability = full_ability.strip()
    matches = list(ABILITY_PREFIX_PATTERN.finditer(full_ability))
    if not matches:
        parts = [full_ability]
    else:
        parts = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(full_ability)
            part = full_ability[match.start() : end].strip()
            if part:
                parts.append(part)

    normalized: list[str] = []
    for part in parts:
        part = part.replace("Non-literal communication:", "Non-Literal Communication:")
        part = SUBABILITY_MERGE_MAP.get(part, part)
        if part not in normalized:
            normalized.append(part)
    return normalized


def load_task_to_subabilities() -> dict[str, dict[int, tuple[str, ...]]]:
    task_to_subabilities: dict[str, dict[int, tuple[str, ...]]] = {}
    data_dir = TOM_ROOT / "data"
    known_subabilities = set(SUBABILITY_ORDER)

    for path in sorted(data_dir.glob("*.jsonl")):
        question_subabilities: dict[int, tuple[str, ...]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for question_idx, line in enumerate(handle):
                record = json.loads(line)
                labels = split_subabilities(record.get("能力\nABILITY", ""))
                unknown = [label for label in labels if label not in known_subabilities]
                if unknown:
                    raise ValueError(f"Unknown ToM-Bench subabilities in {path.name}: {unknown}")
                question_subabilities[question_idx] = tuple(labels)
        task_to_subabilities[path.stem] = question_subabilities

    return task_to_subabilities


def most_common(items: list[str]) -> str:
    return Counter(items).most_common(1)[0][0]


def load_tom_task_stats(result_path: Path) -> tuple[int, int]:
    predictions: dict[int, list[str]] = defaultdict(list)
    answers: dict[int, str] = {}
    with result_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = int(row["question_idx"])
            predictions[qid].append(row["mapped_prediction"])
            answers.setdefault(qid, row["answer"])

    correct = 0
    for qid, preds in predictions.items():
        if most_common(preds) == answers[qid]:
            correct += 1
    total = len(predictions)
    return correct, total


def load_tom_question_outcomes(result_path: Path) -> dict[int, bool]:
    predictions: dict[int, list[str]] = defaultdict(list)
    answers: dict[int, str] = {}
    with result_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            qid = int(row["question_idx"])
            predictions[qid].append(row["mapped_prediction"])
            answers.setdefault(qid, row["answer"])

    outcomes: dict[int, bool] = {}
    for qid, preds in predictions.items():
        outcomes[qid] = most_common(preds) == answers[qid]
    return outcomes


def collect_tom_task_stats(results_dir: Path, variants: tuple[TomVariant, ...]) -> dict[str, dict[str, tuple[int, int]]]:
    task_stats_by_model: dict[str, dict[str, tuple[int, int]]] = {}

    for variant in variants:
        pattern = f"**/*_{variant.result_group}_results.jsonl"
        matched_files = sorted(results_dir.glob(pattern))
        if not matched_files:
            continue

        task_paths: dict[str, Path] = {}
        for path in matched_files:
            task = path.name[: -len(f"_{variant.result_group}_results.jsonl")]
            current = task_paths.get(task)
            if current is None or len(path.parts) > len(current.parts):
                task_paths[task] = path

        task_stats: dict[str, tuple[int, int]] = {}
        for task, path in sorted(task_paths.items()):
            correct, total = load_tom_task_stats(path)
            task_stats[task] = (correct, total)
        task_stats_by_model[variant.model_name] = task_stats

    return task_stats_by_model


def collect_tom_subability_stats(
    results_dir: Path,
    variants: tuple[TomVariant, ...],
    task_to_subabilities: dict[str, dict[int, tuple[str, ...]]],
) -> dict[str, dict[str, tuple[int, int]]]:
    stats_by_model: dict[str, dict[str, tuple[int, int]]] = {}

    for variant in variants:
        pattern = f"**/*_{variant.result_group}_results.jsonl"
        matched_files = sorted(results_dir.glob(pattern))
        if not matched_files:
            continue

        task_paths: dict[str, Path] = {}
        for path in matched_files:
            task = path.name[: -len(f"_{variant.result_group}_results.jsonl")]
            current = task_paths.get(task)
            if current is None or len(path.parts) > len(current.parts):
                task_paths[task] = path

        aggregated: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for task, path in sorted(task_paths.items()):
            question_outcomes = load_tom_question_outcomes(path)
            question_subabilities = task_to_subabilities.get(task)
            if question_subabilities is None:
                continue

            for question_idx, is_correct in question_outcomes.items():
                labels = question_subabilities.get(question_idx, ())
                for label in labels:
                    aggregated[label][0] += int(is_correct)
                    aggregated[label][1] += 1

        stats_by_model[variant.model_name] = {
            label: (correct_total[0], correct_total[1]) for label, correct_total in aggregated.items()
        }

    return stats_by_model


def build_tom_language_values(
    task_stats: dict[str, tuple[int, int]],
    task_to_ability: dict[str, str],
    formatter,
) -> tuple[list[str], list[str]]:
    task_values: list[str] = []
    ability_correct_total: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for task in OFFICIAL_TASK_ORDER:
        stats = task_stats.get(task)
        if not stats:
            task_values.append("N/A")
            continue

        correct, total = stats
        accuracy = correct / total if total else 0.0
        task_values.append(formatter(accuracy, total))

    for task, (correct, total) in sorted(task_stats.items()):
        ability = task_to_ability.get(task)
        if not ability:
            continue
        ability_correct_total[ability][0] += correct
        ability_correct_total[ability][1] += total

    ability_values: list[str] = []
    for ability in ABILITY_ORDER:
        correct, total = ability_correct_total.get(ability, [0, 0])
        if total == 0:
            ability_values.append("N/A")
            continue
        accuracy = correct / total
        ability_values.append(formatter(accuracy, total))

    return task_values, ability_values


def build_tom_tables(config: ModelConfig, task_to_ability: dict[str, str]) -> tuple[list[str], list[list[str]], list[list[str]], list[str]]:
    zh_results_dir = TOM_ROOT / config.tom_folder / "results"
    en_results_dir = TOM_ROOT / config.tom_folder / "results_en"
    benchmark_task_path = TOM_ROOT / config.tom_folder / "tombench_task_summary_with_ci.csv"
    benchmark_ability_path = TOM_ROOT / config.tom_folder / "tombench_ability_summary_with_ci.csv"
    report_task_path = REPORT_ROOT / config.report_dir / "ToM-Bench" / "ToM-Bench_task_summary_with_ci.csv"
    report_ability_path = REPORT_ROOT / config.report_dir / "ToM-Bench" / "ToM-Bench_ability_summary_with_ci.csv"

    zh_task_stats_by_model = collect_tom_task_stats(zh_results_dir, config.tom_variants)
    en_task_stats_by_model = collect_tom_task_stats(en_results_dir, config.tom_variants)
    warnings: list[str] = []
    expected_task_count = len(task_to_ability)

    task_header = bilingual_header(OFFICIAL_TASK_ORDER)
    ability_header = bilingual_header(ABILITY_ORDER)
    task_rows_percent: list[list[str]] = []
    task_rows_decimal: list[list[str]] = []
    ability_rows_percent: list[list[str]] = []
    ability_rows_decimal: list[list[str]] = []

    for variant in config.tom_variants:
        model_name = variant.model_name
        zh_task_stats = zh_task_stats_by_model.get(model_name, {})
        zh_matched_task_count = len(zh_task_stats)

        if zh_matched_task_count == 0:
            if model_name.endswith("LoRA"):
                continue
            warnings.append(f"[ToM-Bench] {config.tom_folder} 缺少 {model_name} 的中文结果文件，未写入新表")
            continue

        if zh_matched_task_count < expected_task_count:
            warnings.append(
                f"[ToM-Bench] {config.tom_folder} 的 {model_name} 中文结果仅覆盖 {zh_matched_task_count}/{expected_task_count} 个 task，未写入新表"
            )
            continue

        zh_task_values_percent, zh_ability_values_percent = build_tom_language_values(
            zh_task_stats, task_to_ability, format_tom_percent
        )
        zh_task_values_decimal, zh_ability_values_decimal = build_tom_language_values(
            zh_task_stats, task_to_ability, format_tom_decimal
        )

        en_task_stats = en_task_stats_by_model.get(model_name, {})
        en_matched_task_count = len(en_task_stats)
        if en_matched_task_count == 0:
            en_task_values_percent = [""] * len(OFFICIAL_TASK_ORDER)
            en_task_values_decimal = [""] * len(OFFICIAL_TASK_ORDER)
            en_ability_values_percent = [""] * len(ABILITY_ORDER)
            en_ability_values_decimal = [""] * len(ABILITY_ORDER)
        elif en_matched_task_count < expected_task_count:
            warnings.append(
                f"[ToM-Bench] {config.tom_folder} 的 {model_name} 英文结果仅覆盖 {en_matched_task_count}/{expected_task_count} 个 task，英文列留空"
            )
            en_task_values_percent = [""] * len(OFFICIAL_TASK_ORDER)
            en_task_values_decimal = [""] * len(OFFICIAL_TASK_ORDER)
            en_ability_values_percent = [""] * len(ABILITY_ORDER)
            en_ability_values_decimal = [""] * len(ABILITY_ORDER)
        else:
            en_task_values_percent, en_ability_values_percent = build_tom_language_values(
                en_task_stats, task_to_ability, format_tom_percent
            )
            en_task_values_decimal, en_ability_values_decimal = build_tom_language_values(
                en_task_stats, task_to_ability, format_tom_decimal
            )

        task_rows_percent.append(bilingual_row(model_name, zh_task_values_percent, en_task_values_percent))
        task_rows_decimal.append(bilingual_row(model_name, zh_task_values_decimal, en_task_values_decimal))
        ability_rows_percent.append(bilingual_row(model_name, zh_ability_values_percent, en_ability_values_percent))
        ability_rows_decimal.append(bilingual_row(model_name, zh_ability_values_decimal, en_ability_values_decimal))

    write_csv(benchmark_task_path, task_header, task_rows_percent)
    write_csv(benchmark_ability_path, ability_header, ability_rows_percent)
    write_csv(report_task_path, task_header, task_rows_decimal)
    write_csv(report_ability_path, ability_header, ability_rows_decimal)

    return ability_header, ability_rows_decimal, task_rows_decimal, warnings


def build_tom_subability_values(
    subability_stats: dict[str, tuple[int, int]],
    formatter,
) -> list[str]:
    values: list[str] = []
    for subability in SUBABILITY_ORDER:
        correct, total = subability_stats.get(subability, (0, 0))
        if total == 0:
            values.append("N/A")
            continue
        accuracy = correct / total
        values.append(formatter(accuracy, total))
    return values


def build_tom_subability_rows(
    config: ModelConfig,
    task_to_subabilities: dict[str, dict[int, tuple[str, ...]]],
) -> tuple[list[list[str]], list[str]]:
    zh_results_dir = TOM_ROOT / config.tom_folder / "results"
    en_results_dir = TOM_ROOT / config.tom_folder / "results_en"

    zh_task_stats_by_model = collect_tom_task_stats(zh_results_dir, config.tom_variants)
    en_task_stats_by_model = collect_tom_task_stats(en_results_dir, config.tom_variants)
    zh_subability_stats_by_model = collect_tom_subability_stats(zh_results_dir, config.tom_variants, task_to_subabilities)
    en_subability_stats_by_model = collect_tom_subability_stats(en_results_dir, config.tom_variants, task_to_subabilities)

    expected_task_count = len(task_to_subabilities)
    rows_decimal: list[list[str]] = []
    warnings: list[str] = []

    for variant in config.tom_variants:
        model_name = variant.model_name
        zh_task_stats = zh_task_stats_by_model.get(model_name, {})
        zh_matched_task_count = len(zh_task_stats)
        if zh_matched_task_count == 0:
            if model_name.endswith("LoRA"):
                continue
            warnings.append(f"[ToM-Bench] {config.tom_folder} 缺少 {model_name} 的中文结果文件，31 子能力表未写入")
            continue
        if zh_matched_task_count < expected_task_count:
            warnings.append(
                f"[ToM-Bench] {config.tom_folder} 的 {model_name} 中文结果仅覆盖 {zh_matched_task_count}/{expected_task_count} 个 task，31 子能力表未写入"
            )
            continue

        zh_values_decimal = build_tom_subability_values(
            zh_subability_stats_by_model.get(model_name, {}),
            format_tom_decimal,
        )

        en_task_stats = en_task_stats_by_model.get(model_name, {})
        en_matched_task_count = len(en_task_stats)
        if en_matched_task_count == 0:
            en_values_decimal = [""] * len(SUBABILITY_ORDER)
        elif en_matched_task_count < expected_task_count:
            warnings.append(
                f"[ToM-Bench] {config.tom_folder} 的 {model_name} 英文结果仅覆盖 {en_matched_task_count}/{expected_task_count} 个 task，31 子能力英文列留空"
            )
            en_values_decimal = [""] * len(SUBABILITY_ORDER)
        else:
            en_values_decimal = build_tom_subability_values(
                en_subability_stats_by_model.get(model_name, {}),
                format_tom_decimal,
            )

        rows_decimal.append(bilingual_row(model_name, zh_values_decimal, en_values_decimal))

    return rows_decimal, warnings


def compare_csv_rows(path: Path, generated_rows: list[list[str]]) -> list[str]:
    if not path.exists():
        return [f"{path} 不存在，已新建"]
    existing = load_csv_rows(path)
    existing_serialized = [list(row.values()) for row in existing]
    if existing_serialized == generated_rows:
        return []
    return [f"{path} 已更新"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tom-only", action="store_true", help="Only sync ToM-Bench outputs")
    args = parser.parse_args()

    task_to_ability = load_task_to_ability()
    task_to_subabilities = load_task_to_subabilities()
    eq_summary_rows: list[list[str]] = []
    tom_summary_rows: list[list[str]] = []
    tom_task_summary_rows: list[list[str]] = []
    tom_subability_summary_rows: list[list[str]] = []
    warnings: list[str] = []
    updates: list[str] = []

    for config in MODEL_CONFIGS:
        if not args.tom_only:
            eq_rows, eq_warnings = build_eq_rows(config)
            warnings.extend(eq_warnings)

            eq_header = ["Model", "Benchmark Score", "Num Questions Parseable", "Run Date"]
            eq_summary_path = EQ_ROOT / config.eq_folder / "eqbench_results_summary.csv"
            report_eq_path = REPORT_ROOT / config.report_dir / "eq-bench" / "eq-bench_results.csv"
            updates.extend(compare_csv_rows(eq_summary_path, eq_rows))
            updates.extend(compare_csv_rows(report_eq_path, eq_rows))
            write_csv(eq_summary_path, eq_header, eq_rows)
            write_csv(report_eq_path, eq_header, eq_rows)
            eq_summary_rows.extend(eq_rows)

        _, ability_rows_decimal, task_rows_decimal, tom_warnings = build_tom_tables(config, task_to_ability)
        subability_rows_decimal, tom_subability_warnings = build_tom_subability_rows(config, task_to_subabilities)
        warnings.extend(tom_warnings)
        warnings.extend(tom_subability_warnings)
        tom_summary_rows.extend(ability_rows_decimal)
        tom_task_summary_rows.extend(task_rows_decimal)
        tom_subability_summary_rows.extend(subability_rows_decimal)

    eq_summary_path = REPORT_SUMMARY_ROOT / "eq-bench.csv"
    tom_summary_path = REPORT_SUMMARY_ROOT / "ToM-Bench_ability.csv"
    tom_task_summary_path = REPORT_SUMMARY_ROOT / "ToM-Bench_task.csv"
    tom_subability_summary_path = REPORT_SUMMARY_ROOT / "ToM-Bench_subability.csv"
    eq_header = ["Model", "Benchmark Score", "Num Questions Parseable", "Run Date"]
    tom_header = bilingual_header(ABILITY_ORDER)
    tom_task_header = bilingual_header(OFFICIAL_TASK_ORDER)
    tom_subability_header = bilingual_header(SUBABILITY_ORDER)

    if not args.tom_only:
        updates.extend(compare_csv_rows(eq_summary_path, eq_summary_rows))
    updates.extend(compare_csv_rows(tom_summary_path, tom_summary_rows))
    updates.extend(compare_csv_rows(tom_task_summary_path, tom_task_summary_rows))
    updates.extend(compare_csv_rows(tom_subability_summary_path, tom_subability_summary_rows))
    if not args.tom_only:
        write_csv(eq_summary_path, eq_header, eq_summary_rows)
    write_csv(tom_summary_path, tom_header, tom_summary_rows)
    write_csv(tom_task_summary_path, tom_task_header, tom_task_summary_rows)
    write_csv(tom_subability_summary_path, tom_subability_header, tom_subability_summary_rows)

    print("同步完成。")
    if updates:
        print("更新文件:")
        for item in sorted(set(updates)):
            print(f"  - {item}")
    if warnings:
        print("注意:")
        for item in warnings:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
