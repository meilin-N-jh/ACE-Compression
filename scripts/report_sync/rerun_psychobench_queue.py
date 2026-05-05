#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import http.client
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PB_ROOT = ROOT / "paper_benchmarks" / "10_PsychoBench" / "benchmark"
RERUN_CSV = ROOT / "report_summary" / "Psychobench_needs_rerun.csv"
LOCAL_NO_PROXY = "127.0.0.1,localhost"
QUESTIONNAIRE_ORDER = [
    "BFI",
    "DTDD",
    "EPQ-R",
    "ECR-R",
    "CABIN",
    "GSE",
    "LMS",
    "BSRI",
    "ICB",
    "LOT-R",
    "Empathy",
    "EIS",
    "WLEIS",
    "16P",
]
QUESTIONNAIRE_RANK = {name: index for index, name in enumerate(QUESTIONNAIRE_ORDER)}


@dataclass(frozen=True)
class TaskSpec:
    run_name: str
    group: str
    runner: Path
    result_dir: Path
    model_name: str
    model_path: Path
    served_model_name: str
    port: int
    server_args: tuple[str, ...]
    awq: bool


@dataclass(frozen=True)
class Task:
    spec: TaskSpec
    questionnaires: tuple[str, ...]


@dataclass
class TaskOutcome:
    run_name: str
    gpu: int
    success: bool
    questionnaires: tuple[str, ...]
    phase: str
    server_log: Path
    run_log: Path
    detail: str


def print_now(lock: threading.Lock, message: str) -> None:
    with lock:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)


def ordered_questionnaires(names: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    deduped.sort(key=lambda item: (QUESTIONNAIRE_RANK.get(item, 999), item))
    return tuple(deduped)


def build_specs() -> dict[str, TaskSpec]:
    specs: dict[str, TaskSpec] = {}

    def add(
        run_name: str,
        group: str,
        runner: str,
        result_dir: str,
        model_name: str,
        model_path: str,
        served_model_name: str,
        port: int,
        server_args: tuple[str, ...],
        awq: bool = False,
    ) -> None:
        specs[run_name] = TaskSpec(
            run_name=run_name,
            group=group,
            runner=ROOT / runner,
            result_dir=ROOT / result_dir,
            model_name=model_name,
            model_path=ROOT / model_path,
            served_model_name=served_model_name,
            port=port,
            server_args=server_args,
            awq=awq,
        )

    float16_args = (
        "--max-model-len", "8192",
        "--dtype", "float16",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.9",
        "--disable-log-stats",
    )
    auto_heavy_args = (
        "--max-model-len", "8192",
        "--dtype", "auto",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.98",
        "--max-num-seqs", "1024",
        "--max-num-batched-tokens", "32768",
        "--disable-log-stats",
    )
    deepseek_auto_args = (
        "--max-model-len", "8192",
        "--dtype", "auto",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.95",
        "--max-num-seqs", "1024",
        "--max-num-batched-tokens", "32768",
        "--disable-log-stats",
    )
    deepseek_fp16_args = (
        "--max-model-len", "8192",
        "--dtype", "auto",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "1024",
        "--max-num-batched-tokens", "32768",
        "--disable-log-stats",
    )
    llama_auto_args = (
        "--max-model-len", "8192",
        "--dtype", "auto",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.9",
        "--disable-log-stats",
    )

    add(
        "Qwen2.5-7B-2:4-Sparse",
        "qwen2.5-7b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/run_psychobench_qwen25_7b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/results/Qwen2.5-7B-2:4-Sparse",
        "qwen2.5-7b-2to4-sparse",
        "models/Qwen2.5-7B/Qwen2.5-7B-Instruct-2to4-Sparse",
        "qwen2.5-7b-2to4-sparse",
        8404,
        float16_args,
    )
    add(
        "Qwen2.5-7B-BNB-4bit",
        "qwen2.5-7b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/run_psychobench_qwen25_7b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/results/Qwen2.5-7B-BNB-4bit",
        "qwen2.5-7b-bnb-4bit",
        "models/Qwen2.5-7B/Qwen2.5-7B-Instruct",
        "qwen2.5-7b-bnb-4bit",
        8401,
        (
            "--max-model-len", "8192",
            "--dtype", "auto",
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.98",
            "--max-num-seqs", "1024",
            "--max-num-batched-tokens", "32768",
            "--disable-log-stats",
        ),
    )
    add(
        "Qwen2.5-7B-GPTQ-INT4",
        "qwen2.5-7b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/run_psychobench_qwen25_7b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/results/Qwen2.5-7B-GPTQ-INT4",
        "qwen2.5-7b-gptq-int4",
        "models/Qwen2.5-7B/Qwen2.5-7B-Instruct-GPTQ-Int4",
        "qwen2.5-7b-gptq-int4",
        8403,
        auto_heavy_args,
    )
    add(
        "Qwen2.5-7B-Unstructured-Sparse",
        "qwen2.5-7b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/run_psychobench_qwen25_7b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/results/Qwen2.5-7B-Unstructured-Sparse",
        "qwen2.5-7b-unstructured-sparse",
        "models/Qwen2.5-7B/Qwen2.5-7B-Instruct-Unstructured-Sparse-50",
        "qwen2.5-7b-unstructured-sparse",
        8405,
        float16_args,
    )

    add(
        "Qwen2.5-14B-2:4-Sparse",
        "qwen2.5-14b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/run_psychobench_qwen25_14b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/results/Qwen2.5-14B-2:4-Sparse",
        "qwen2.5-14b-2to4-sparse",
        "models/Qwen2.5-14B/Qwen2.5-14B-2to4-Sparse",
        "qwen2.5-14b-2to4-sparse",
        8105,
        float16_args,
    )
    add(
        "Qwen2.5-14B-AWQ",
        "qwen2.5-14b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/run_psychobench_qwen25_14b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/results/Qwen2.5-14B-AWQ",
        "qwen2.5-14b-awq",
        "models/Qwen2.5-14B/Qwen2.5-14B-Instruct-AWQ",
        "qwen2.5-14b-awq",
        8102,
        (
            "--quantization", "awq",
            "--host", "127.0.0.1",
        ),
        awq=True,
    )
    add(
        "Qwen2.5-14B-BNB-4bit",
        "qwen2.5-14b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/run_psychobench_qwen25_14b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/results/Qwen2.5-14B-BNB-4bit",
        "qwen2.5-14b-bnb-4bit",
        "models/Qwen2.5-14B/Qwen2.5-14B-Instruct-FP16",
        "qwen2.5-14b-bnb-4bit",
        8101,
        (
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
            "--max-model-len", "8192",
            "--dtype", "auto",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.98",
            "--max-num-seqs", "1024",
            "--max-num-batched-tokens", "32768",
            "--disable-log-stats",
        ),
    )
    add(
        "Qwen2.5-14B-GPTQ-INT4",
        "qwen2.5-14b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/run_psychobench_qwen25_14b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/results/Qwen2.5-14B-GPTQ-INT4",
        "qwen2.5-14b-gptq-int4",
        "models/Qwen2.5-14B/Qwen2.5-14B-Instruct-GPTQ-Int4",
        "qwen2.5-14b-gptq-int4",
        8103,
        (
            "--quantization", "gptq",
            "--max-model-len", "8192",
            "--dtype", "auto",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.98",
            "--max-num-seqs", "1024",
            "--max-num-batched-tokens", "32768",
            "--disable-log-stats",
        ),
    )
    add(
        "Qwen2.5-14B-Unstructured-Sparse",
        "qwen2.5-14b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/run_psychobench_qwen25_14b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/results/Qwen2.5-14B-Unstructured-Sparse",
        "qwen2.5-14b-unstructured-sparse",
        "models/Qwen2.5-14B/Qwen2.5-14B-Unstructured-Sparse-50",
        "qwen2.5-14b-unstructured-sparse",
        8106,
        float16_args,
    )

    add(
        "Qwen2.5-32B-2:4-Sparse",
        "qwen2.5-32b",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-32b/run_psychobench_qwen25_32b.py",
        "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-32b/results/Qwen2.5-32B-2:4-Sparse",
        "qwen2.5-32b-2to4-sparse",
        "models/Qwen2.5-32B/Qwen2.5-32B-2to4-Sparse",
        "qwen2.5-32b-2to4-sparse",
        8208,
        float16_args,
    )

    add(
        "Llama3.1-8B-2:4-Sparse",
        "llama3.1-8b",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/run_psychobench_llama.py",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/results/Llama3.1-8B-2:4-Sparse",
        "meta-llama-3.1-8b-instruct-2to4-sparse",
        "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-2to4-Sparse",
        "meta-llama-3.1-8b-instruct-2to4-sparse",
        8410,
        float16_args,
    )
    add(
        "Llama3.1-8B-AWQ",
        "llama3.1-8b",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/run_psychobench_llama.py",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/results/Llama3.1-8B-AWQ",
        "meta-llama-3.1-8b-instruct-awq-int4",
        "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        "meta-llama-3.1-8b-instruct-awq-int4",
        8402,
        llama_auto_args,
        awq=True,
    )
    add(
        "Llama3.1-8B-BNB-4bit",
        "llama3.1-8b",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/run_psychobench_llama.py",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/results/Llama3.1-8B-BNB-4bit",
        "meta-llama-3.1-8b-instruct-bnb-4bit",
        "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct",
        "meta-llama-3.1-8b-instruct-bnb-4bit",
        8401,
        (
            "--max-model-len", "8192",
            "--dtype", "auto",
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.9",
            "--disable-log-stats",
        ),
    )
    add(
        "Llama3.1-8B-FP16",
        "llama3.1-8b",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/run_psychobench_llama.py",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/results/Llama3.1-8B-FP16",
        "meta-llama-3.1-8b-instruct-fp16",
        "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct",
        "meta-llama-3.1-8b-instruct-fp16",
        8400,
        float16_args,
    )
    add(
        "Llama3.1-8B-GPTQ-INT4",
        "llama3.1-8b",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/run_psychobench_llama.py",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/results/Llama3.1-8B-GPTQ-INT4",
        "meta-llama-3.1-8b-instruct-gptq-int4",
        "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",
        "meta-llama-3.1-8b-instruct-gptq-int4",
        8403,
        llama_auto_args,
    )
    add(
        "Llama3.1-8B-Unstructured-Sparse",
        "llama3.1-8b",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/run_psychobench_llama.py",
        "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/results/Llama3.1-8B-Unstructured-Sparse",
        "meta-llama-3.1-8b-instruct-unstructured-sparse",
        "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse-50",
        "meta-llama-3.1-8b-instruct-unstructured-sparse",
        8411,
        float16_args,
    )

    add(
        "DeepSeek-V2-Lite-16B-2:4-Sparse",
        "deepseek-v2-lite-16b",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/run_psychobench_deepseek.py",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/results/DeepSeek-V2-Lite-16B-2:4-Sparse",
        "deepseek-v2-lite-2to4-sparse",
        "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-2to4-Sparse",
        "deepseek-v2-lite-2to4-sparse",
        8304,
        float16_args,
    )
    add(
        "DeepSeek-V2-Lite-16B-AWQ",
        "deepseek-v2-lite-16b",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/run_psychobench_deepseek.py",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/results/DeepSeek-V2-Lite-16B-AWQ",
        "deepseek-v2-lite-awq",
        "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-AWQ",
        "deepseek-v2-lite-awq",
        8302,
        deepseek_auto_args,
        awq=True,
    )
    add(
        "DeepSeek-V2-Lite-16B-BNB-4bit",
        "deepseek-v2-lite-16b",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/run_psychobench_deepseek.py",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/results/DeepSeek-V2-Lite-16B-BNB-4bit",
        "deepseek-v2-lite-bnb-4bit",
        "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat",
        "deepseek-v2-lite-bnb-4bit",
        8301,
        (
            "--quantization", "bitsandbytes",
            "--load-format", "bitsandbytes",
            "--max-model-len", "8192",
            "--dtype", "auto",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.95",
            "--max-num-seqs", "1024",
            "--max-num-batched-tokens", "32768",
            "--disable-log-stats",
        ),
    )
    add(
        "DeepSeek-V2-Lite-16B-FP16",
        "deepseek-v2-lite-16b",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/run_psychobench_deepseek.py",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/results/DeepSeek-V2-Lite-16B-FP16",
        "deepseek-v2-lite-fp16",
        "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat",
        "deepseek-v2-lite-fp16",
        8300,
        deepseek_fp16_args,
    )
    add(
        "DeepSeek-V2-Lite-16B-Unstructured-Sparse",
        "deepseek-v2-lite-16b",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/run_psychobench_deepseek.py",
        "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/results/DeepSeek-V2-Lite-16B-Unstructured-Sparse",
        "deepseek-v2-lite-unstructured-sparse",
        "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-Unstructured-Sparse-50",
        "deepseek-v2-lite-unstructured-sparse",
        8305,
        float16_args,
    )

    for spec in specs.values():
        if not spec.runner.exists():
            raise FileNotFoundError(f"Missing runner: {spec.runner}")
        if not spec.model_path.exists():
            raise FileNotFoundError(f"Missing model path: {spec.model_path}")

    qwen14_awq = specs["Qwen2.5-14B-AWQ"]
    specs["Qwen2.5-14B-AWQ"] = TaskSpec(
        run_name=qwen14_awq.run_name,
        group=qwen14_awq.group,
        runner=qwen14_awq.runner,
        result_dir=qwen14_awq.result_dir,
        model_name=qwen14_awq.model_name,
        model_path=qwen14_awq.model_path,
        served_model_name=qwen14_awq.served_model_name,
        port=qwen14_awq.port,
        server_args=(
            "--quantization", "awq",
            "--max-model-len", "8192",
            "--dtype", "auto",
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.98",
            "--max-num-seqs", "1024",
            "--max-num-batched-tokens", "32768",
            "--disable-log-stats",
        ),
        awq=True,
    )

    return specs


def load_tasks(specs: dict[str, TaskSpec], only: set[str] | None) -> list[Task]:
    grouped: dict[str, list[str]] = defaultdict(list)
    if RERUN_CSV.exists():
        with RERUN_CSV.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                variant = row["variant"]
                if only is not None and variant not in only:
                    continue
                grouped[variant].append(row["questionnaire"])
    else:
        if only is None:
            raise FileNotFoundError(f"Missing rerun manifest: {RERUN_CSV}")

        run_logs_root = PB_ROOT / "rerun_logs"
        outcome_paths = sorted(
            run_logs_root.glob("psychobench_rerun_*/outcomes.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        pending = set(only)
        for outcome_path in outcome_paths:
            with outcome_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    variant = row.get("run_name", "").strip()
                    if variant not in pending:
                        continue
                    questionnaires = [
                        item.strip()
                        for item in row.get("questionnaires", "").split(",")
                        if item.strip()
                    ]
                    if not questionnaires:
                        continue
                    grouped[variant].extend(questionnaires)
                    pending.remove(variant)
                    if not pending:
                        break
            if not pending:
                break

        if grouped:
            print(
                f"Rerun manifest missing at {RERUN_CSV}; "
                "reconstructed tasks from latest outcomes logs",
                flush=True,
            )
        if pending:
            missing = ", ".join(sorted(pending))
            raise FileNotFoundError(
                f"Missing rerun manifest {RERUN_CSV} and no matching outcomes found for: {missing}"
            )

    tasks: list[Task] = []
    for variant, questionnaires in sorted(grouped.items()):
        if variant not in specs:
            raise KeyError(f"No task spec for {variant}")
        tasks.append(Task(spec=specs[variant], questionnaires=ordered_questionnaires(questionnaires)))
    return tasks


def make_target_paths(task: Task) -> list[Path]:
    result_dir = task.spec.result_dir
    paths: list[Path] = []
    for questionnaire in task.questionnaires:
        stem = f"{task.spec.run_name}-{questionnaire}"
        paths.append(result_dir / f"{stem}.csv")
        paths.append(result_dir / f"{stem}.md")
        paths.append(result_dir / f"{stem}.png")
        paths.append(result_dir / "prompts" / f"{stem}-shuffle0.txt")
        paths.append(result_dir / "responses" / f"{stem}-shuffle0.txt")
    return paths


def backup_targets(task: Task, backup_root: Path, fresh_csv: bool = False) -> dict[Path, Path | None]:
    manifest: dict[Path, Path | None] = {}
    for path in make_target_paths(task):
        if path.exists():
            rel = path.relative_to(task.spec.result_dir)
            backup_path = backup_root / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_path)
            manifest[path] = backup_path
        else:
            manifest[path] = None

    purge_suffixes = {".txt", ".png"}
    if fresh_csv:
        purge_suffixes.update({".csv", ".md"})

    for path in make_target_paths(task):
        if path.suffix in purge_suffixes and path.exists():
            path.unlink()

    return manifest


def restore_targets(manifest: dict[Path, Path | None]) -> None:
    for path, backup in manifest.items():
        if backup is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)


def terminate_process(proc: subprocess.Popen[bytes] | None, timeout: int = 30) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def wait_for_server(
    port: int,
    timeout_s: int = 900,
    proc: subprocess.Popen[bytes] | None = None,
) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with opener.open(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError):
            pass
        time.sleep(5)
    return False


def tail_text(path: Path, limit: int = 40) -> str:
    if not path.exists():
        return f"{path} does not exist"
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[-limit:])


def verify_task(task: Task, start_time: float) -> list[str]:
    issues: list[str] = []
    for questionnaire in task.questionnaires:
        stem = f"{task.spec.run_name}-{questionnaire}"
        for path in (
            task.spec.result_dir / f"{stem}.csv",
            task.spec.result_dir / f"{stem}.md",
            task.spec.result_dir / "prompts" / f"{stem}-shuffle0.txt",
            task.spec.result_dir / "responses" / f"{stem}-shuffle0.txt",
        ):
            if not path.exists():
                issues.append(f"missing {path}")
                continue
            if path.stat().st_mtime < start_time:
                issues.append(f"stale {path}")
    return issues


def start_server(task: Task, gpu: int, server_log: Path) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["NO_PROXY"] = LOCAL_NO_PROXY
    env["no_proxy"] = LOCAL_NO_PROXY
    cmd = [
        "conda", "run", "-n", "qwen2.5",
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(task.spec.model_path),
        "--served-model-name", task.spec.served_model_name,
        "--host", "127.0.0.1",
        "--port", str(task.spec.port),
        *task.spec.server_args,
    ]
    handle = server_log.open("wb")
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def run_benchmark(task: Task, run_log: Path) -> int:
    base_url = f"http://127.0.0.1:{task.spec.port}/v1"
    env = os.environ.copy()
    env["NO_PROXY"] = LOCAL_NO_PROXY
    env["no_proxy"] = LOCAL_NO_PROXY
    cmd = [
        "conda", "run", "-n", "qwen2.5",
        "python", "-u", str(task.spec.runner),
        "--base-url", base_url,
        "--model", task.spec.model_name,
        "--questionnaire", ",".join(task.questionnaires),
        "--shuffle-count", "1",
        "--test-count", "10",
        "--name-exp", task.spec.run_name,
        "--significance-level", "0.01",
        "--mode", "auto",
    ]
    with run_log.open("wb") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return proc.wait()


def run_task(
    task: Task,
    gpu: int,
    phase: str,
    run_root: Path,
    port_lock: threading.Lock,
    print_lock: threading.Lock,
    fresh_csv: bool,
) -> TaskOutcome:
    safe_name = task.spec.run_name.replace(":", "_")
    server_log = run_root / f"{safe_name}.server.log"
    run_log = run_root / f"{safe_name}.run.log"
    backup_root = run_root / "backups" / safe_name
    manifest = backup_targets(task, backup_root, fresh_csv=fresh_csv)
    server_proc: subprocess.Popen[bytes] | None = None

    with port_lock:
        try:
            print_now(
                print_lock,
                f"GPU {gpu} start {task.spec.run_name} on port {task.spec.port} "
                f"for {','.join(task.questionnaires)}",
            )
            server_proc = start_server(task, gpu, server_log)
            if not wait_for_server(task.spec.port, proc=server_proc):
                detail = f"server did not become ready\n{tail_text(server_log)}"
                restore_targets(manifest)
                return TaskOutcome(
                    run_name=task.spec.run_name,
                    gpu=gpu,
                    success=False,
                    questionnaires=task.questionnaires,
                    phase=phase,
                    server_log=server_log,
                    run_log=run_log,
                    detail=detail,
                )

            print_now(print_lock, f"GPU {gpu} running benchmark for {task.spec.run_name}")
            start_time = time.time()
            exit_code = run_benchmark(task, run_log)
            if exit_code != 0:
                detail = f"benchmark exit code {exit_code}\n{tail_text(run_log)}"
                restore_targets(manifest)
                return TaskOutcome(
                    run_name=task.spec.run_name,
                    gpu=gpu,
                    success=False,
                    questionnaires=task.questionnaires,
                    phase=phase,
                    server_log=server_log,
                    run_log=run_log,
                    detail=detail,
                )

            issues = verify_task(task, start_time)
            if issues:
                detail = "\n".join(issues[-20:])
                restore_targets(manifest)
                return TaskOutcome(
                    run_name=task.spec.run_name,
                    gpu=gpu,
                    success=False,
                    questionnaires=task.questionnaires,
                    phase=phase,
                    server_log=server_log,
                    run_log=run_log,
                    detail=detail,
                )

            print_now(print_lock, f"GPU {gpu} finished {task.spec.run_name}")
            return TaskOutcome(
                run_name=task.spec.run_name,
                gpu=gpu,
                success=True,
                questionnaires=task.questionnaires,
                phase=phase,
                server_log=server_log,
                run_log=run_log,
                detail="ok",
            )
        finally:
            terminate_process(server_proc)
            time.sleep(3)


def worker_loop(
    gpu: int,
    phase: str,
    task_queue: queue.Queue[Task],
    outcomes: list[TaskOutcome],
    outcomes_lock: threading.Lock,
    port_locks: dict[int, threading.Lock],
    run_root: Path,
    print_lock: threading.Lock,
    fresh_csv: bool,
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return
        try:
            outcome = run_task(
                task,
                gpu,
                phase,
                run_root,
                port_locks[task.spec.port],
                print_lock,
                fresh_csv,
            )
        except Exception as exc:  # pragma: no cover - defensive guard for long runs
            safe_name = task.spec.run_name.replace(":", "_")
            outcome = TaskOutcome(
                run_name=task.spec.run_name,
                gpu=gpu,
                success=False,
                questionnaires=task.questionnaires,
                phase=phase,
                server_log=run_root / f"{safe_name}.server.log",
                run_log=run_root / f"{safe_name}.run.log",
                detail=f"worker exception: {exc}",
            )
            print_now(print_lock, f"GPU {gpu} crashed on {task.spec.run_name}: {exc}")
        with outcomes_lock:
            outcomes.append(outcome)
        task_queue.task_done()


def run_phase(
    phase_name: str,
    tasks: list[Task],
    gpus: list[int],
    run_root: Path,
    print_lock: threading.Lock,
    fresh_csv: bool,
) -> list[TaskOutcome]:
    if not tasks:
        return []
    print_now(print_lock, f"Phase {phase_name} starting with {len(tasks)} tasks on GPUs {gpus}")
    task_queue: queue.Queue[Task] = queue.Queue()
    for task in sorted(tasks, key=lambda item: len(item.questionnaires), reverse=True):
        task_queue.put(task)

    outcomes: list[TaskOutcome] = []
    outcomes_lock = threading.Lock()
    port_locks = defaultdict(threading.Lock)
    threads = [
        threading.Thread(
            target=worker_loop,
            args=(
                gpu,
                phase_name,
                task_queue,
                outcomes,
                outcomes_lock,
                port_locks,
                run_root,
                print_lock,
                fresh_csv,
            ),
            daemon=True,
        )
        for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if len(outcomes) != len(tasks):
        raise RuntimeError(
            f"Phase {phase_name} produced {len(outcomes)} outcomes for {len(tasks)} tasks"
        )
    print_now(
        print_lock,
        f"Phase {phase_name} finished: "
        f"{sum(1 for outcome in outcomes if outcome.success)}/{len(outcomes)} tasks succeeded",
    )
    return outcomes


def write_outcomes(path: Path, outcomes: list[TaskOutcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "phase",
            "run_name",
            "gpu",
            "success",
            "questionnaires",
            "server_log",
            "run_log",
            "detail",
        ])
        for outcome in sorted(outcomes, key=lambda item: (item.phase, item.run_name)):
            writer.writerow([
                outcome.phase,
                outcome.run_name,
                outcome.gpu,
                "yes" if outcome.success else "no",
                ",".join(outcome.questionnaires),
                str(outcome.server_log),
                str(outcome.run_log),
                outcome.detail,
            ])


def run_command_logged(cmd: list[str], log_path: Path, print_lock: threading.Lock) -> None:
    print_now(print_lock, f"Postprocess: {' '.join(cmd)}")
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def run_postprocess(run_root: Path, print_lock: threading.Lock) -> None:
    commands = [
        ["python3", "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-7b/extract_results.py"],
        ["python3", "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-14b/extract_results.py"],
        ["python3", "paper_benchmarks/10_PsychoBench/benchmark/qwen2.5-32b/extract_results.py"],
        ["python3", "paper_benchmarks/10_PsychoBench/benchmark/llama3.1-8b/extract_results.py"],
        ["python3", "paper_benchmarks/10_PsychoBench/benchmark/deepseek-v2-lite-16b/extract_results.py"],
        ["python3", "scripts/sync_psychobench_reports.py"],
        ["python3", "scripts/validate_psychobench_summary.py"],
    ]
    for index, cmd in enumerate(commands, start=1):
        log_path = run_root / f"postprocess_{index}.log"
        run_command_logged(cmd, log_path, print_lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun problematic PsychoBench questionnaires.")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6", help="Comma-separated GPU ids.")
    parser.add_argument("--only", default="", help="Comma-separated run names to limit execution.")
    parser.add_argument(
        "--fresh-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild target csv/md from scratch before rerun. Use --no-fresh-csv to reuse existing files.",
    )
    parser.add_argument("--skip-postprocess", action="store_true", help="Skip extract/sync/validate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("No GPUs specified")

    only = {item.strip() for item in args.only.split(",") if item.strip()} or None
    specs = build_specs()
    tasks = load_tasks(specs, only)
    if not tasks:
        print("No tasks selected", flush=True)
        return 0

    run_root = PB_ROOT / "rerun_logs" / time.strftime("psychobench_rerun_%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    print_lock = threading.Lock()

    print_now(print_lock, f"Selected {len(tasks)} tasks")
    print_now(
        print_lock,
        f"GPUs={','.join(str(gpu) for gpu in gpus)} fresh_csv={'yes' if args.fresh_csv else 'no'}",
    )
    for task in tasks:
        print_now(print_lock, f"  {task.spec.run_name}: {','.join(task.questionnaires)}")

    non_awq = [task for task in tasks if not task.spec.awq]
    awq = [task for task in tasks if task.spec.awq]

    outcomes: list[TaskOutcome] = []
    outcomes.extend(run_phase("non_awq", non_awq, gpus, run_root, print_lock, args.fresh_csv))
    outcomes.extend(run_phase("awq", awq, gpus, run_root, print_lock, args.fresh_csv))

    outcomes_path = run_root / "outcomes.csv"
    write_outcomes(outcomes_path, outcomes)
    failures = [outcome for outcome in outcomes if not outcome.success]

    if failures:
        print_now(print_lock, f"{len(failures)} tasks failed; see {outcomes_path}")
        for outcome in failures:
            print_now(print_lock, f"FAILED {outcome.run_name}: {outcome.detail}")
        return 1

    if not args.skip_postprocess:
        run_postprocess(run_root, print_lock)

    print_now(print_lock, f"All tasks completed successfully. Outcomes: {outcomes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
