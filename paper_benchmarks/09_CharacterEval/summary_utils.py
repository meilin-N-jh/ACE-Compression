#!/usr/bin/env python3
"""Shared utilities for generating official-aligned CharacterEval summaries."""

import csv
import json
from collections import defaultdict
from pathlib import Path


OFFICIAL_METRICS = [
    "Accuracy",
    "Behavior",
    "Coherence",
    "Communication_skills",
    "Consistency",
    "Diversity",
    "Empathy",
    "Exposure",
    "Fluency",
    "Hallucination",
    "Humanlikeness",
    "Utterance",
]


def compute_scores(eval_file: Path) -> dict[str, float]:
    with open(eval_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    score_dict: dict[str, list[float]] = defaultdict(list)
    for record in records:
        metric = record.get("metric_en")
        if metric in OFFICIAL_METRICS and metric in record:
            score_dict[metric].append(float(record[metric]))

    scores = {}
    for metric in OFFICIAL_METRICS:
        metric_scores = score_dict.get(metric, [])
        if metric_scores:
            scores[metric] = sum(metric_scores) / len(metric_scores)

    if scores:
        scores["Total"] = sum(scores[metric] for metric in OFFICIAL_METRICS if metric in scores) / len(scores)

    return scores


def discover_models(results_dir: Path) -> list[str]:
    models = []
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        eval_file = model_dir / f"{model_dir.name}_evaluation.jsonl"
        if eval_file.exists():
            models.append(model_dir.name)
    return models


def write_summary_csv(
    results_dir: Path,
    output_file: Path,
    include_series: bool = False,
    series_name: str | None = None,
    exclude_models: set[str] | None = None,
) -> list[dict[str, str]]:
    models = discover_models(results_dir)
    if exclude_models:
        models = [model for model in models if model not in exclude_models]
    rows: list[dict[str, str]] = []

    for model in models:
        eval_file = results_dir / model / f"{model}_evaluation.jsonl"
        scores = compute_scores(eval_file)
        row = {"Model": model}
        if include_series:
            row["Series"] = series_name or results_dir.parent.name
        for metric in ["Total", *OFFICIAL_METRICS]:
            row[metric] = f"{scores[metric]:.4f}" if metric in scores else "N/A"
        rows.append(row)

    fieldnames = ["Series", "Model", "Total", *OFFICIAL_METRICS] if include_series else ["Model", "Total", *OFFICIAL_METRICS]
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows
