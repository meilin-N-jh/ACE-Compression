#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
TOMBENCH_ROOT = ROOT / "paper_benchmarks" / "07_ToM-Bench" / "benchmark"
ALL_TASKS = [
    line.strip().removeprefix("data/").removesuffix(".jsonl")
    for line in (TOMBENCH_ROOT / "all_tasks.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


def is_missing(value: object) -> bool:
    return value is None or value == "" or (isinstance(value, float) and value != value)


def normalize_choices(choices: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in choices.items() if not is_missing(value)}


def local_api_env(base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    return env


@dataclass(frozen=True)
class VariantSpec:
    display_name: str
    served_model_name: str
    model_path: str
    server_env: str
    gpu: str | None = None
    port: int | None = None
    vllm_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroupSpec:
    name: str
    folder: str
    runner_script: str
    runner_env: str
    gpu: str
    port: int
    variants: tuple[VariantSpec, ...]

    @property
    def benchmark_dir(self) -> Path:
        return TOMBENCH_ROOT / self.folder

    @property
    def results_dir(self) -> Path:
        return self.benchmark_dir / "results_en"

    @property
    def logs_dir(self) -> Path:
        return self.benchmark_dir / "logs_en"

    @property
    def runner_path(self) -> Path:
        return self.benchmark_dir / self.runner_script


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        name="qwen2.5-7b",
        folder="qwen2.5-7b",
        runner_script="run_tombench_qwen25_vllm.py",
        runner_env="qwen2.5",
        gpu="0",
        port=8400,
        variants=(
            VariantSpec("Qwen2.5-7B-FP16", "qwen2.5-7b-fp16", str(ROOT / "models/Qwen2.5-7B/Qwen2.5-7B-Instruct"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("Qwen2.5-7B-BNB-4bit", "qwen2.5-7b-bnb-4bit", str(ROOT / "models/Qwen2.5-7B/Qwen2.5-7B-Instruct"), "qwen2.5",
                        vllm_args=("--quantization", "bitsandbytes", "--load-format", "bitsandbytes", "--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("Qwen2.5-7B-GPTQ-INT4", "qwen2.5-7b-gptq-int4", str(ROOT / "models/Qwen2.5-7B/Qwen2.5-7B-Instruct-GPTQ-Int4"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("Qwen2.5-7B-2:4-Sparse", "qwen2.5-7b-2to4-sparse", str(ROOT / "models/Qwen2.5-7B/Qwen2.5-7B-Instruct-2to4-Sparse"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Qwen2.5-7B-Unstructured-Sparse", "qwen2.5-7b-unstructured-sparse", str(ROOT / "models/Qwen2.5-7B/Qwen2.5-7B-Instruct-Unstructured-Sparse-50"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Qwen2.5-7B-AWQ", "qwen2.5-7b-awq", str(ROOT / "models/Qwen2.5-7B/Qwen2.5-7B-Instruct-AWQ"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
        ),
    ),
    GroupSpec(
        name="qwen2.5-14b",
        folder="qwen2.5-14b",
        runner_script="run_tombench_qwen25_vllm.py",
        runner_env="qwen2.5",
        gpu="1",
        port=8100,
        variants=(
            VariantSpec("Qwen2.5-14B-FP16", "qwen2.5-14b-fp16", str(ROOT / "models/Qwen2.5-14B/Qwen2.5-14B-Instruct-FP16"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("Qwen2.5-14B-BNB-4bit", "qwen2.5-14b-bnb-4bit", str(ROOT / "models/Qwen2.5-14B/Qwen2.5-14B-Instruct-FP16"), "qwen2.5",
                        vllm_args=("--quantization", "bitsandbytes", "--load-format", "bitsandbytes", "--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("Qwen2.5-14B-GPTQ-INT4", "qwen2.5-14b-gptq-int4", str(ROOT / "models/Qwen2.5-14B/Qwen2.5-14B-Instruct-GPTQ-Int4"), "qwen2.5",
                        vllm_args=("--quantization", "gptq", "--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("Qwen2.5-14B-Trim", "qwen2.5-14b-trim", str(ROOT / "models/Qwen2.5-14B/qwen2.5-14b-instruct-trim"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("Qwen2.5-14B-2:4-Sparse", "qwen2.5-14b-2to4-sparse", str(ROOT / "models/Qwen2.5-14B/Qwen2.5-14B-2to4-Sparse"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Qwen2.5-14B-Unstructured-Sparse", "qwen2.5-14b-unstructured-sparse", str(ROOT / "models/Qwen2.5-14B/Qwen2.5-14B-Unstructured-Sparse-50"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Qwen2.5-14B-AWQ", "qwen2.5-14b-awq", str(ROOT / "models/Qwen2.5-14B/Qwen2.5-14B-Instruct-AWQ"), "qwen2.5",
                        vllm_args=("--quantization", "awq", "--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.98", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
        ),
    ),
    GroupSpec(
        name="deepseek-v2-lite-16b",
        folder="deepseek-v2-lite-16b",
        runner_script="run_tombench_deepseek_vllm.py",
        runner_env="qwen2.5",
        gpu="2",
        port=8300,
        variants=(
            VariantSpec("DeepSeek-V2-Lite-16B-FP16", "deepseek-v2-lite-fp16", str(ROOT / "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("DeepSeek-V2-Lite-16B-BNB-4bit", "deepseek-v2-lite-bnb-4bit", str(ROOT / "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat"), "qwen2.5",
                        vllm_args=("--quantization", "bitsandbytes", "--load-format", "bitsandbytes", "--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.95", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
            VariantSpec("DeepSeek-V2-Lite-16B-2:4-Sparse", "deepseek-v2-lite-2to4-sparse", str(ROOT / "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-2to4-Sparse"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("DeepSeek-V2-Lite-16B-Unstructured-Sparse", "deepseek-v2-lite-unstructured-sparse", str(ROOT / "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-Unstructured-Sparse-50"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("DeepSeek-V2-Lite-16B-AWQ", "deepseek-v2-lite-awq", str(ROOT / "models/DeepSeek-V2-Lite-Chat-16B/DeepSeek-V2-Lite-Chat-AWQ"), "qwen2.5",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.95", "--max-num-seqs", "1024", "--max-num-batched-tokens", "32768", "--disable-log-stats")),
        ),
    ),
    GroupSpec(
        name="qwen2.5-32b",
        folder="qwen2.5-32b",
        runner_script="run_tombench_qwen25_32b_vllm.py",
        runner_env="qwen2.5",
        gpu="3",
        port=8207,
        variants=(
            VariantSpec("Qwen2.5-32B-FP16", "qwen2.5-32b-fp16", str(ROOT / "models/Qwen2.5-32B/Qwen2.5-32B-Instruct"), "qwen2.5",
                        vllm_args=("--max-model-len", "4096", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.95", "--max-num-seqs", "64", "--max-num-batched-tokens", "4096", "--disable-log-stats")),
            VariantSpec("Qwen2.5-32B-BNB-4bit", "qwen2.5-32b-bnb-4bit", str(ROOT / "models/Qwen2.5-32B/Qwen2.5-32B-Instruct"), "qwen2.5",
                        vllm_args=("--quantization", "bitsandbytes", "--load-format", "bitsandbytes", "--max-model-len", "4096", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "128", "--max-num-batched-tokens", "8192", "--disable-log-stats")),
            VariantSpec("Qwen2.5-32B-GPTQ-INT4", "qwen2.5-32b-gptq-int4", str(ROOT / "models/Qwen2.5-32B/Qwen2.5-32B-Instruct-GPTQ-Int4"), "qwen2.5",
                        vllm_args=("--quantization", "gptq", "--max-model-len", "4096", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "128", "--max-num-batched-tokens", "8192", "--disable-log-stats")),
            VariantSpec("Qwen2.5-32B-2:4-Sparse", "qwen2.5-32b-2to4-sparse", str(ROOT / "models/Qwen2.5-32B/Qwen2.5-32B-2to4-Sparse"), "qwen2.5",
                        vllm_args=("--max-model-len", "4096", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "64", "--max-num-batched-tokens", "4096", "--disable-log-stats")),
            VariantSpec("Qwen2.5-32B-Unstructured-Sparse", "qwen2.5-32b-unstructured-sparse", str(ROOT / "models/Qwen2.5-32B/Qwen2.5-32B-Unstructured-Sparse-50"), "qwen2.5",
                        vllm_args=("--max-model-len", "4096", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "64", "--max-num-batched-tokens", "4096", "--disable-log-stats")),
            VariantSpec("Qwen2.5-32B-AWQ", "qwen2.5-32b-awq", str(ROOT / "models/Qwen2.5-32B/Qwen2.5-32B-Instruct-AWQ"), "qwen2.5",
                        vllm_args=("--quantization", "awq", "--max-model-len", "4096", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.90", "--max-num-seqs", "128", "--max-num-batched-tokens", "8192", "--disable-log-stats")),
        ),
    ),
    GroupSpec(
        name="llama3.1-8b",
        folder="llama3.1-8b",
        runner_script="run_tombench_llama_vllm.py",
        runner_env="llama",
        gpu="4",
        port=8412,
        variants=(
            VariantSpec("Llama3.1-8B-FP16", "meta-llama-3.1-8b-instruct-fp16", str(ROOT / "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct"), "llama",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Llama3.1-8B-BNB-4bit", "meta-llama-3.1-8b-instruct-bnb-4bit", str(ROOT / "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct"), "llama",
                        vllm_args=("--quantization", "bitsandbytes", "--load-format", "bitsandbytes", "--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Llama3.1-8B-GPTQ-INT4", "meta-llama-3.1-8b-instruct-gptq-int4", str(ROOT / "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"), "llama",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Llama3.1-8B-2:4-Sparse", "meta-llama-3.1-8b-instruct-2to4-sparse", str(ROOT / "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-2to4-Sparse"), "llama",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Llama3.1-8B-Unstructured-Sparse", "meta-llama-3.1-8b-instruct-unstructured-sparse", str(ROOT / "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-Unstructured-Sparse-50"), "llama",
                        vllm_args=("--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Llama3.1-8B-AWQ", "meta-llama-3.1-8b-instruct-awq-int4", str(ROOT / "models/llama3.1-8b/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"), "llama",
                        vllm_args=("--max-model-len", "8192", "--dtype", "auto", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
        ),
    ),
    GroupSpec(
        name="llama70B",
        folder="llama70B",
        runner_script="run_tombench_llama70b_vllm.py",
        runner_env="llama",
        gpu="5,6",
        port=8000,
        variants=(
            VariantSpec("Llama-70B-FP16", "llama31-70b-fp16", str(ROOT / "models/Llama3-70b/Llama-3.1-70B-Instruct"), "llama", gpu="5,6", port=8000,
                        vllm_args=("--tensor-parallel-size", "2", "--max-model-len", "4096", "--dtype", "float16", "--trust-remote-code", "--disable-log-stats")),
            VariantSpec("Llama70B-2:4-Sparse", "llama31-70b-2to4-sparse", str(ROOT / "models/Llama3-70b/Llama-3.1-70B-Instruct-2to4-Sparse"), "llama", gpu="5,6", port=8010,
                        vllm_args=("--tensor-parallel-size", "2", "--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Llama70B-Unstructured-Sparse", "llama31-70b-unstructured-sparse", str(ROOT / "models/Llama3-70b/Llama-3.1-70B-Instruct-Unstructured-Sparse-50"), "llama", gpu="5,6", port=8011,
                        vllm_args=("--tensor-parallel-size", "2", "--max-model-len", "8192", "--dtype", "float16", "--trust-remote-code", "--gpu-memory-utilization", "0.9", "--disable-log-stats")),
            VariantSpec("Llama-70B-4bit", "llama31-70b-bnb4bit", str(ROOT / "models/Llama3-70b/Llama-3.1-70B-Instruct"), "llama", gpu="5", port=8006,
                        vllm_args=("--quantization", "bitsandbytes", "--max-model-len", "4096", "--dtype", "float16", "--trust-remote-code", "--disable-log-stats")),
            VariantSpec("Llama-70B-GPTQ-INT4", "llama31-70b-gptq-int4", str(ROOT / "models/Llama3-70b/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4"), "llama", gpu="6", port=8004,
                        vllm_args=("--quantization", "gptq", "--max-model-len", "4096", "--dtype", "float16", "--trust-remote-code", "--disable-log-stats")),
            VariantSpec("Llama-70B-GPTQ-INT8", "llama31-70b-gptq-int8", str(ROOT / "models/Llama3-70b/Meta-Llama-3.1-70B-Instruct-GPTQ-INT8"), "llama", gpu="5", port=8007,
                        vllm_args=("--quantization", "gptq", "--max-model-len", "4096", "--dtype", "float16", "--trust-remote-code", "--disable-log-stats")),
            VariantSpec("Llama-70B-AWQ", "llama31-70b-awq-int4", str(ROOT / "models/Llama3-70b/Llama-3.1-70B-Instruct-AWQ-INT4"), "awq", gpu="6", port=8001,
                        vllm_args=("--quantization", "awq", "--tensor-parallel-size", "1", "--max-model-len", "4096", "--dtype", "float16", "--trust-remote-code", "--disable-log-stats")),
        ),
    ),
)


def wait_for_server(port: int, timeout_s: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()
    session = requests.Session()
    session.trust_env = False
    while time.time() - start < timeout_s:
        try:
            response = session.get(url, timeout=5)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"vLLM server on port {port} did not become ready within {timeout_s}s")


def localized_fields(d: dict, language: str) -> tuple[str, str, dict[str, str]]:
    if language == "en":
        story = d.get("STORY", d.get("故事", ""))
        question = d.get("QUESTION", d.get("问题", ""))
        choices = {
            "A": d.get("OPTION-A", d.get("选项A", "")),
            "B": d.get("OPTION-B", d.get("选项B", "")),
        }
        if not is_missing(d.get("OPTION-C", d.get("选项C"))):
            choices["C"] = d.get("OPTION-C", d.get("选项C", ""))
        if not is_missing(d.get("OPTION-D", d.get("选项D"))):
            choices["D"] = d.get("OPTION-D", d.get("选项D", ""))
        return story, question, choices

    story = d.get("故事", d.get("STORY", ""))
    question = d.get("问题", d.get("QUESTION", ""))
    choices = {
        "A": d.get("选项A", d.get("OPTION-A", "")),
        "B": d.get("选项B", d.get("OPTION-B", "")),
    }
    if not is_missing(d.get("选项C", d.get("OPTION-C"))):
        choices["C"] = d.get("选项C", d.get("OPTION-C", ""))
    if not is_missing(d.get("选项D", d.get("OPTION-D"))):
        choices["D"] = d.get("选项D", d.get("OPTION-D", ""))
    return story, question, choices


def stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def close_process_log(proc: subprocess.Popen[bytes] | None) -> None:
    handle = getattr(proc, "_codex_log_handle", None)
    if handle is not None:
        handle.close()


def validate_result_language(results_dir: Path, variant: VariantSpec, task: str, language: str) -> bool:
    result_path = results_dir / f"{task}_{variant.display_name}_results.jsonl"
    source_path = TOMBENCH_ROOT / "data" / f"{task}.jsonl"
    try:
        source_line = source_path.read_text(encoding="utf-8").splitlines()[0]
        result_line = result_path.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError):
        return False

    try:
        source_record = json.loads(source_line)
        result_record = json.loads(result_line)
    except json.JSONDecodeError:
        return False

    story, question, choices = localized_fields(source_record, language)
    return (
        result_record.get("language") == language
        and result_record.get("story") == story
        and result_record.get("question") == question
        and normalize_choices(result_record.get("choices", {})) == normalize_choices(choices)
    )


def has_complete_results(results_dir: Path, variant: VariantSpec, language: str) -> bool:
    if not results_dir.exists():
        return False
    for task in ALL_TASKS:
        path = results_dir / f"{task}_{variant.display_name}_results.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            return False
        if not validate_result_language(results_dir, variant, task, language):
            return False
    return True


def start_server(group: GroupSpec, variant: VariantSpec, server_log: Path, conda_env_override: str | None) -> subprocess.Popen[bytes]:
    port = variant.port if variant.port is not None else group.port
    server_env = conda_env_override or variant.server_env
    cmd = [
        "conda", "run", "-n", server_env, "python", "-u",
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", variant.model_path,
        "--served-model-name", variant.served_model_name,
        "--host", "127.0.0.1",
        "--port", str(port),
        *variant.vllm_args,
    ]
    env = local_api_env()
    env["CUDA_VISIBLE_DEVICES"] = variant.gpu if variant.gpu is not None else group.gpu
    env["PYTHONUNBUFFERED"] = "1"
    handle = server_log.open("wb")
    proc = subprocess.Popen(
        cmd,
        stdout=handle,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        env=env,
        start_new_session=True,
    )
    proc._codex_log_handle = handle  # type: ignore[attr-defined]
    return proc


def run_benchmark(
    group: GroupSpec,
    variant: VariantSpec,
    try_times: int,
    seed: int,
    bench_log: Path,
    conda_env_override: str | None,
) -> None:
    env = local_api_env()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOMBENCH_RESULTS_DIR"] = str(group.results_dir)
    runner_env = conda_env_override or group.runner_env
    cmd = [
        "conda", "run", "-n", runner_env, "python3", str(group.runner_path),
        "--model-display-name", variant.display_name,
        "--base-url", f"http://127.0.0.1:{variant.port if variant.port is not None else group.port}",
        "--model-name", variant.served_model_name,
        "--language", "en",
        "--try-times", str(try_times),
        "--seed", str(seed),
    ]
    with bench_log.open("wb") as handle:
        subprocess.run(cmd, cwd=str(group.benchmark_dir), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def run_variant(
    group: GroupSpec,
    variant: VariantSpec,
    try_times: int,
    seed: int,
    force: bool,
    server_timeout: int,
    conda_env_override: str | None,
) -> None:
    group.results_dir.mkdir(parents=True, exist_ok=True)
    group.logs_dir.mkdir(parents=True, exist_ok=True)

    if not force and has_complete_results(group.results_dir, variant, language="en"):
        print(f"[skip] {group.name} / {variant.display_name}")
        return

    server_log = group.logs_dir / f"vllm_{variant.display_name}.log"
    bench_log = group.logs_dir / f"eval_{variant.display_name}.log"
    proc: subprocess.Popen[bytes] | None = None
    gpu = variant.gpu if variant.gpu is not None else group.gpu
    port = variant.port if variant.port is not None else group.port
    try:
        print(f"[start] {group.name} / {variant.display_name} on GPU {gpu} port {port}")
        proc = start_server(group, variant, server_log, conda_env_override=conda_env_override)
        wait_for_server(port, timeout_s=server_timeout)
        print(f"[ready] {group.name} / {variant.display_name}")
        run_benchmark(
            group,
            variant,
            try_times=try_times,
            seed=seed,
            bench_log=bench_log,
            conda_env_override=conda_env_override,
        )
        print(f"[done] {group.name} / {variant.display_name}")
    finally:
        stop_process(proc)
        close_process_log(proc)
        print(f"[stop] {group.name} / {variant.display_name}")


def run_group(
    group: GroupSpec,
    try_times: int,
    seed: int,
    force: bool,
    server_timeout: int,
    conda_env_override: str | None,
) -> None:
    print(f"[group] {group.name} -> GPU {group.gpu}")
    if group.name != "llama70B":
        for variant in group.variants:
            run_variant(
                group,
                variant,
                try_times=try_times,
                seed=seed,
                force=force,
                server_timeout=server_timeout,
                conda_env_override=conda_env_override,
            )
        return

    for variant in group.variants[:3]:
        run_variant(
            group,
            variant,
            try_times=try_times,
            seed=seed,
            force=force,
            server_timeout=server_timeout,
            conda_env_override=conda_env_override,
        )

    for wave in (group.variants[3:5], group.variants[5:7]):
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = [
                executor.submit(run_variant, group, variant, try_times, seed, force, server_timeout, conda_env_override)
                for variant in wave
            ]
            for future in as_completed(futures):
                future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one ToM-Bench model group sequentially in English")
    parser.add_argument("--group", required=True, choices=[group.name for group in GROUPS])
    parser.add_argument("--try-times", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--server-timeout", type=int, default=3600)
    parser.add_argument("--conda-env-override", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    group = next(group for group in GROUPS if group.name == args.group)
    run_group(
        group,
        try_times=args.try_times,
        seed=args.seed,
        force=args.force,
        server_timeout=args.server_timeout,
        conda_env_override=args.conda_env_override,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
