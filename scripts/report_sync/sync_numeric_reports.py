from __future__ import annotations
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())


import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(f"{ARTIFACT_ROOT}")
REPORT_ROOT = ROOT / "report"
REPORT_SUMMARY_ROOT = ROOT / "report_summary"


@dataclass(frozen=True)
class VariantConfig:
    key: str
    display_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class GroupConfig:
    key: str
    report_dir: str
    human_report_dir: str
    ceval_dir: str
    roleeval_dir: str
    variants: tuple[VariantConfig, ...]


COMMON_VARIANTS_QWEN_7B = (
    VariantConfig("fp16", "Qwen2.5-7B-Instruct-FP16", ("fp16",)),
    VariantConfig("bnb_4bit", "Qwen2.5-7B-Instruct-BNB-4bit", ("bnb_4bit",)),
    VariantConfig("awq", "Qwen2.5-7B-Instruct-AWQ", ("awq",)),
    VariantConfig("gptq_int4", "Qwen2.5-7B-Instruct-GPTQ-INT4", ("gptq_int4",)),
    VariantConfig("2_4_sparse", "Qwen2.5-7B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
    VariantConfig(
        "unstructured_sparse",
        "Qwen2.5-7B-Instruct-Unstructured-Sparse",
        ("unstructured_sparse",),
    ),
)

COMMON_VARIANTS_QWEN_14B = (
    VariantConfig("fp16", "Qwen2.5-14B-Instruct-FP16", ("fp16",)),
    VariantConfig("bnb_4bit", "Qwen2.5-14B-Instruct-BNB-4bit", ("bnb_4bit",)),
    VariantConfig("awq", "Qwen2.5-14B-Instruct-AWQ", ("awq",)),
    VariantConfig("gptq_int4", "Qwen2.5-14B-Instruct-GPTQ-INT4", ("gptq_int4",)),
    VariantConfig("trim", "Qwen2.5-14B-Instruct-Trim", ("trim",)),
    VariantConfig("2_4_sparse", "Qwen2.5-14B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
    VariantConfig(
        "unstructured_sparse",
        "Qwen2.5-14B-Instruct-Unstructured-Sparse",
        ("unstructured_sparse",),
    ),
)

COMMON_VARIANTS_QWEN_32B = (
    VariantConfig("fp16", "Qwen2.5-32B-Instruct-FP16", ("fp16",)),
    VariantConfig("bnb_4bit", "Qwen2.5-32B-Instruct-BNB-4bit", ("bnb_4bit",)),
    VariantConfig("awq", "Qwen2.5-32B-Instruct-AWQ", ("awq",)),
    VariantConfig("gptq_int4", "Qwen2.5-32B-Instruct-GPTQ-INT4", ("gptq_int4",)),
    VariantConfig("2_4_sparse", "Qwen2.5-32B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
    VariantConfig(
        "unstructured_sparse",
        "Qwen2.5-32B-Instruct-Unstructured-Sparse",
        ("unstructured_sparse",),
    ),
)

COMMON_VARIANTS_DEEPSEEK = (
    VariantConfig("fp16", "DeepSeek-V2-Lite-16B-FP16", ("fp16",)),
    VariantConfig("bnb_4bit", "DeepSeek-V2-Lite-16B-BnB-4bit", ("bnb_4bit",)),
    VariantConfig("awq", "DeepSeek-V2-Lite-16B-AWQ", ("awq",)),
    VariantConfig("2_4_sparse", "DeepSeek-V2-Lite-16B-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
    VariantConfig(
        "unstructured_sparse",
        "DeepSeek-V2-Lite-16B-Unstructured-Sparse",
        ("unstructured_sparse",),
    ),
)

COMMON_VARIANTS_LLAMA8 = (
    VariantConfig("fp16", "Llama3.1-8B-Instruct-FP16", ("fp16",)),
    VariantConfig("bnb_4bit", "Llama3.1-8B-Instruct-BNB-4bit", ("bnb_4bit", "4bit")),
    VariantConfig("awq", "Llama3.1-8B-Instruct-AWQ", ("awq",)),
    VariantConfig("gptq_int4", "Llama3.1-8B-Instruct-GPTQ", ("gptq_int4", "gptq")),
    VariantConfig("2_4_sparse", "Llama3.1-8B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
    VariantConfig(
        "unstructured_sparse",
        "Llama3.1-8B-Instruct-Unstructured-Sparse",
        ("unstructured_sparse",),
    ),
)

COMMON_VARIANTS_LLAMA70 = (
    VariantConfig("fp16", "Llama-3.1-70B-Instruct-FP16", ("fp16",)),
    VariantConfig("bnb_4bit", "Llama-3.1-70B-Instruct-BNB-4bit", ("bnb_4bit", "4bit")),
    VariantConfig("awq", "Llama-3.1-70B-Instruct-AWQ-INT4", ("awq",)),
    VariantConfig("gptq_int4", "Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", ("gptq_int4", "gptq")),
    VariantConfig("gptq_int8", "Meta-Llama-3.1-70B-Instruct-GPTQ-INT8", ("gptq_int8",)),
    VariantConfig("2_4_sparse", "Llama-3.1-70B-Instruct-2:4-Sparse", ("2_4_sparse", "2to4_sparse")),
    VariantConfig(
        "unstructured_sparse",
        "Llama-3.1-70B-Instruct-Unstructured-Sparse",
        ("unstructured_sparse",),
    ),
)

GROUPS = (
    GroupConfig(
        key="qwen2.5-7b",
        report_dir="qwen2.5-7b",
        human_report_dir="qwen2.5-7b",
        ceval_dir="qwen2.5-7b",
        roleeval_dir="qwen2.5-7b",
        variants=COMMON_VARIANTS_QWEN_7B,
    ),
    GroupConfig(
        key="qwen2.5-14b",
        report_dir="qwen2.5-14b",
        human_report_dir="qwen2.5-14b",
        ceval_dir="qwen2.5-14b",
        roleeval_dir="qwen2.5-14b",
        variants=COMMON_VARIANTS_QWEN_14B,
    ),
    GroupConfig(
        key="qwen2.5-32b",
        report_dir="qwen2.5-32b",
        human_report_dir="qwen2.5-32b",
        ceval_dir="qwen2.5-32b",
        roleeval_dir="qwen2.5-32b",
        variants=COMMON_VARIANTS_QWEN_32B,
    ),
    GroupConfig(
        key="deepseek-v2-lite-16b",
        report_dir="deepseek-v2-lite-16b",
        human_report_dir="deepseek-v2-lite-16b",
        ceval_dir="deepseek-v2-lite-16b",
        roleeval_dir="deepseek-v2-lite-16b",
        variants=COMMON_VARIANTS_DEEPSEEK,
    ),
    GroupConfig(
        key="llama3.1-8b",
        report_dir="llama3.1-8b",
        human_report_dir="llama3.1-8b",
        ceval_dir="llama3.1-8b",
        roleeval_dir="llama3.1-8b",
        variants=COMMON_VARIANTS_LLAMA8,
    ),
    GroupConfig(
        key="llama70B",
        report_dir="llama3.1-70b",
        human_report_dir="llama70B",
        ceval_dir="llama70B",
        roleeval_dir="llama3.1-70b",
        variants=COMMON_VARIANTS_LLAMA70,
    ),
)

HUMAN_HEADER = [
    "Model",
    "Pass@1 (Greedy)",
    "Pass@1 (Sampling, n=20)",
    "Pass@10 (Sampling, n=20)",
    "Pass@1 (Sampling, n=100)",
    "Pass@10 (Sampling, n=100)",
    "Pass@100 (Sampling, n=100)",
]
CEVAL_HEADER = ["Model Name", "Valid Acc", "Valid Stderr", "Test Acc", "Test Stderr"]
ROLE_HEADER = [
    "Model",
    "0_en_chinese",
    "0_en_global",
    "0_zh_chinese",
    "0_zh_global",
    "5_en_chinese",
    "5_en_global",
    "5_zh_chinese",
    "5_zh_global",
]


def latest_match(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: (path.name, str(path)))


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def format_float(value: float | int | str) -> str:
    return f"{float(value):.4f}"


def format_pm(value: float | int, stderr: float | int) -> str:
    return f"{float(value):.4f}±{float(stderr):.4f}"


def path_matches_alias(path: Path, aliases: tuple[str, ...]) -> bool:
    name = path.name.lower()
    return any(alias.lower() in name for alias in aliases)


def normalize_alias_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def ceval_payload_matches_alias(payload: dict, aliases: tuple[str, ...]) -> bool:
    config = payload.get("config", {})
    model_args = config.get("model_args")
    candidates: list[str] = []

    if isinstance(model_args, str):
        candidates.append(model_args)
    elif isinstance(model_args, dict):
        for key in (
            "pretrained",
            "model",
            "model_name_or_path",
            "served_model_name",
            "path",
        ):
            value = model_args.get(key)
            if isinstance(value, str):
                candidates.append(value)

    model = config.get("model")
    if isinstance(model, str):
        candidates.append(model)

    haystack = normalize_alias_text(" ".join(candidates))
    normalized_aliases = [normalize_alias_text(alias) for alias in aliases]
    return any(alias in haystack for alias in normalized_aliases)


def collect_humaneval_file(results_dir: Path, aliases: tuple[str, ...], suffix: str) -> Path | None:
    matches: list[Path] = []
    for path in results_dir.glob("*.json"):
        name = path.name.lower()
        if not path_matches_alias(path, aliases):
            continue
        if suffix == "greedy":
            if "pass10" in name or "pass100" in name:
                continue
        elif suffix == "pass10":
            if "pass10" not in name:
                continue
        elif suffix == "pass100":
            if "pass100" not in name:
                continue
        matches.append(path)
    return latest_match(matches)


def build_humaneval_rows(config: GroupConfig) -> list[list[str]]:
    results_dir = ROOT / "paper_benchmarks" / "03_human-eval" / "benchmark" / config.key / "results"
    rows: list[list[str]] = []
    for variant in config.variants:
        greedy = collect_humaneval_file(results_dir, variant.aliases, "greedy")
        pass10 = collect_humaneval_file(results_dir, variant.aliases, "pass10")
        pass100 = collect_humaneval_file(results_dir, variant.aliases, "pass100")
        if not greedy or not pass10 or not pass100:
            continue

        greedy_result = next(iter(load_json(greedy)["results"].values()))
        pass10_result = next(iter(load_json(pass10)["results"].values()))
        pass100_result = next(iter(load_json(pass100)["results"].values()))

        rows.append(
            [
                variant.display_name,
                format_pm(greedy_result["pass@1,create_test"], greedy_result["pass@1_stderr,create_test"]),
                format_pm(pass10_result["pass@1,create_test"], pass10_result["pass@1_stderr,create_test"]),
                format_pm(pass10_result["pass@10,create_test"], pass10_result["pass@10_stderr,create_test"]),
                format_pm(pass100_result["pass@1,create_test"], pass100_result["pass@1_stderr,create_test"]),
                format_pm(pass100_result["pass@10,create_test"], pass100_result["pass@10_stderr,create_test"]),
                format_pm(pass100_result["pass@100,create_test"], pass100_result["pass@100_stderr,create_test"]),
            ]
        )
    return rows


def collect_ceval_json(group_dir: Path, aliases: tuple[str, ...], task_name: str) -> Path | None:
    matches: list[Path] = []
    for candidate in group_dir.rglob("*.json"):
        try:
            payload = load_json(candidate)
        except json.JSONDecodeError:
            continue
        if not (
            path_matches_alias(candidate, aliases)
            or ceval_payload_matches_alias(payload, aliases)
        ):
            continue
        results = payload.get("results", {})
        if task_name not in results:
            continue
        matches.append(candidate)
    return latest_match(matches)


def build_ceval_rows(config: GroupConfig) -> list[list[str]]:
    group_dir = ROOT / "paper_benchmarks" / "05_C-eval" / "launchers" / config.ceval_dir
    rows: list[list[str]] = []
    for variant in config.variants:
        valid_path = collect_ceval_json(group_dir, variant.aliases, "ceval-valid")
        test_path = collect_ceval_json(group_dir, variant.aliases, "ceval-test")
        if not valid_path or not test_path:
            continue
        valid_result = load_json(valid_path)["results"]["ceval-valid"]
        test_result = load_json(test_path)["results"]["ceval-test"]
        rows.append(
            [
                variant.display_name,
                format_float(valid_result["acc,none"]),
                format_float(valid_result["acc_stderr,none"]),
                format_float(test_result["acc,none"]),
                format_float(test_result["acc_stderr,none"]),
            ]
        )
    return rows


def collect_roleeval_json(results_dir: Path, aliases: tuple[str, ...], shot: str, task: str) -> Path | None:
    matches: list[Path] = []
    for path in results_dir.glob("*.json"):
        name = path.name.lower()
        if not path_matches_alias(path, aliases):
            continue
        if f"{shot}shot" not in name or task not in name:
            continue
        matches.append(path)
    return latest_match(matches)


def build_roleeval_rows(config: GroupConfig) -> list[list[str]]:
    results_dir = ROOT / "paper_benchmarks" / "06_roleeval" / "launchers" / config.roleeval_dir / "results"
    rows: list[list[str]] = []
    task_columns = [
        ("0", "en_chinese"),
        ("0", "en_global"),
        ("0", "zh_chinese"),
        ("0", "zh_global"),
        ("5", "en_chinese"),
        ("5", "en_global"),
        ("5", "zh_chinese"),
        ("5", "zh_global"),
    ]

    for variant in config.variants:
        values: list[str] = []
        missing = False
        for shot, task in task_columns:
            path = collect_roleeval_json(results_dir, variant.aliases, shot, task)
            if not path:
                missing = True
                break
            result_key = f"roleeval_{task}"
            result = load_json(path)["results"][result_key]["acc,none"]
            values.append(str(result))
        if missing:
            continue
        rows.append([variant.display_name, *values])
    return rows


def sync_humaneval() -> None:
    summary_rows: list[list[str]] = []
    for config in GROUPS:
        rows = build_humaneval_rows(config)
        benchmark_path = ROOT / "paper_benchmarks" / "03_human-eval" / "benchmark" / config.key / "humaneval_results_summary.csv"
        report_path = REPORT_ROOT / config.human_report_dir / "human-eval" / "human-eval_results.csv"
        write_csv(benchmark_path, HUMAN_HEADER, rows)
        write_csv(report_path, HUMAN_HEADER, rows)
        summary_rows.extend(rows)
    write_csv(REPORT_SUMMARY_ROOT / "human-eval.csv", HUMAN_HEADER, summary_rows)


def sync_ceval() -> None:
    summary_rows: list[list[str]] = []
    for config in GROUPS:
        rows = build_ceval_rows(config)
        benchmark_path = ROOT / "paper_benchmarks" / "05_C-eval" / "launchers" / config.ceval_dir / "ceval_summary.csv"
        report_path = REPORT_ROOT / config.report_dir / "C-eval" / "C-eval_results.csv"
        write_csv(benchmark_path, CEVAL_HEADER, rows)
        write_csv(report_path, CEVAL_HEADER, rows)
        summary_rows.extend(rows)
    write_csv(REPORT_SUMMARY_ROOT / "C-eval.csv", CEVAL_HEADER, summary_rows)


def sync_roleeval() -> None:
    summary_rows: list[list[str]] = []
    for config in GROUPS:
        rows = build_roleeval_rows(config)
        benchmark_path = ROOT / "paper_benchmarks" / "06_roleeval" / "launchers" / config.roleeval_dir / "roleeval_results_summary.csv"
        report_path = REPORT_ROOT / config.report_dir / "role-eval" / "role-eval_results.csv"
        write_csv(benchmark_path, ROLE_HEADER, rows)
        write_csv(report_path, ROLE_HEADER, rows)
        summary_rows.extend(rows)
    write_csv(REPORT_SUMMARY_ROOT / "role-eval.csv", ROLE_HEADER, summary_rows)


def main() -> None:
    sync_humaneval()
    sync_ceval()
    sync_roleeval()


if __name__ == "__main__":
    main()
