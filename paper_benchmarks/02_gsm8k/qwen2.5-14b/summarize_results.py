#!/usr/bin/env python3
"""
Summarize GSM8K evaluation results for all Qwen2.5-14B models
"""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())


import json
import os
from pathlib import Path

RESULTS_DIR = f"{ARTIFACT_ROOT}/paper_benchmarks/02_gsm8k/launchers/qwen2.5-14b/results"
OUTPUT_CSV = f"{ARTIFACT_ROOT}/paper_benchmarks/02_gsm8k/launchers/qwen2.5-14b/gsm8k_results_summary.csv"

# Model display names mapping
MODEL_NAMES = {
    "fp16": "Qwen2.5-14B-Instruct-FP16",
    "bnb_4bit": "Qwen2.5-14B-Instruct-BNB-4bit",
    "awq": "Qwen2.5-14B-Instruct-AWQ",
    "gptq_int4": "Qwen2.5-14B-Instruct-GPTQ-INT4",
    "trim": "Qwen2.5-14B-Instruct-Trim"
}

def extract_results(json_file):
    """Extract results from a JSON file"""
    with open(json_file) as f:
        data = json.load(f)

    results = data.get("results", {})
    if "gsm8k_cot" in results:
        task_results = results["gsm8k_cot"]
        return {
            "strict_match": task_results.get("exact_match,strict-match", None),
            "strict_match_stderr": task_results.get("exact_match_stderr,strict-match", None),
            "flexible_match": task_results.get("exact_match,flexible-extract", None),
            "flexible_match_stderr": task_results.get("exact_match_stderr,flexible-extract", None)
        }
    return None

def main():
    # Find all result JSON files
    result_files = sorted(Path(RESULTS_DIR).glob("gsm8k_cot_*_vllm_*.json"))

    if not result_files:
        print(f"No result files found in {RESULTS_DIR}")
        return

    print(f"Found {len(result_files)} result files")
    print("=" * 80)

    # Extract results for each model
    all_results = {}
    for json_file in result_files:
        # Extract model type from filename
        filename = json_file.name
        if "fp16" in filename:
            model_type = "fp16"
        elif "bnb_4bit" in filename:
            model_type = "bnb_4bit"
        elif "awq" in filename:
            model_type = "awq"
        elif "gptq_int4" in filename:
            model_type = "gptq_int4"
        elif "trim" in filename:
            model_type = "trim"
        else:
            continue

        results = extract_results(json_file)
        if results:
            all_results[model_type] = results
            print(f"✓ {MODEL_NAMES[model_type]}")
            print(f"  Strict Match: {results['strict_match']:.2%}")
            print(f"  Flexible Match: {results['flexible_match']:.2%}")
        else:
            print(f"✗ Failed to extract: {filename}")

    # Generate CSV
    if all_results:
        print("\n" + "=" * 80)
        print("Generating CSV summary...")
        print("=" * 80)

        with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
            # Write header
            f.write("Model,Strict Match,Strict Match StdErr,Flexible Match,Flexible Match StdErr\n")

            # Write results for each model
            for model_type in ["fp16", "bnb_4bit", "awq", "gptq_int4", "trim"]:
                if model_type in all_results:
                    r = all_results[model_type]
                    model_name = MODEL_NAMES[model_type]
                    strict = f"{r['strict_match']:.4f}" if r['strict_match'] is not None else "N/A"
                    strict_err = f"{r['strict_match_stderr']:.4f}" if r['strict_match_stderr'] is not None else "N/A"
                    flexible = f"{r['flexible_match']:.4f}" if r['flexible_match'] is not None else "N/A"
                    flexible_err = f"{r['flexible_match_stderr']:.4f}" if r['flexible_match_stderr'] is not None else "N/A"

                    f.write(f"{model_name},{strict},{strict_err},{flexible},{flexible_err}\n")

        print(f"\n✓ CSV summary saved to: {OUTPUT_CSV}")

        # Also print summary to console
        print("\n" + "=" * 80)
        print("GSM8K RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Model':<40} {'Strict Match':>15} {'Flexible Match':>15}")
        print("-" * 80)
        for model_type in ["fp16", "bnb_4bit", "awq", "gptq_int4", "trim"]:
            if model_type in all_results:
                r = all_results[model_type]
                model_name = MODEL_NAMES[model_type]
                strict = f"{r['strict_match']:.2%}" if r['strict_match'] is not None else "N/A"
                flexible = f"{r['flexible_match']:.2%}" if r['flexible_match'] is not None else "N/A"
                print(f"{model_name:<40} {strict:>15} {flexible:>15}")
        print("=" * 80)

if __name__ == "__main__":
    main()
