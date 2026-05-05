#!/usr/bin/env python3
"""RoleEval bilingual evaluation for DeepSeek-V2-Lite-16B variants (zh + en, chinese + global)."""
import os
import sys
import argparse

# 禁用输出缓冲，确保实时显示进度
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

ROLE_EVAL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROLE_EVAL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from providers import TransformersProvider, AutoGPTQProvider, LlamaCppProvider, VLLMProvider
from metrics import extract_answer, calculate_accuracy, print_metrics

DATA_ROOT = ROLE_EVAL_ROOT / "upstream"


def load_config(config_path: Path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_samples(lang: str, split: str, subset: str, max_samples: int | None = None, categories: list[str] | None = None):
    """Load samples from {lang}/{subset}/{split} (dev/test)."""
    subset_dir = DATA_ROOT / lang / subset / split
    csv_files = sorted(subset_dir.glob(f"*_{split}.csv"))

    if categories:
        categories_set = {c.strip() for c in categories if c.strip()}
        csv_files = [
            f for f in csv_files
            if f.stem.replace(f"_{split}", "") in categories_set
        ]

    samples = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        category = csv_file.stem.replace(f"_{split}", "")
        for _, row in df.iterrows():
            samples.append({
                "subset": subset,
                "split": split,
                "category": category,
                "id": row.get("id", ""),
                "question": row.get("question", ""),
                "A": row.get("A", ""),
                "B": row.get("B", ""),
                "C": row.get("C", ""),
                "D": row.get("D", ""),
                "answer": row.get("answer", ""),
            })

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def format_prompt(sample: dict, template: str) -> str:
    return template.format(
        question=sample["question"],
        option_a=sample["A"],
        option_b=sample["B"],
        option_c=sample["C"],
        option_d=sample["D"],
    )


def init_provider(model_config: dict):
    model_type = model_config.get("model_type")
    if model_type == "transformers":
        return TransformersProvider(model_config)
    if model_type == "autogptq":
        return AutoGPTQProvider(model_config)
    if model_type == "llamacpp":
        return LlamaCppProvider(model_config)
    if model_type == "vllm":
        return VLLMProvider(model_config)
    raise ValueError(f"Unknown model_type: {model_type}")


def run_eval(model_key: str, config: dict, lang: str, subset: str, split: str, output_dir: Path, max_samples: int | None, categories: list[str] | None):
    model_config = dict(config.get("generation_config", {}))
    model_config.update(config["models"][model_key])
    prompt_template = config["prompt_template"]

    print(f"\n{'='*60}")
    print(f"Model: {model_key}")
    print(f"Lang: {lang} | Subset: {subset} | Split: {split}")
    model_path = model_config.get("model_path", "(vLLM remote)")
    print(f"Model path: {model_path}")
    print(f"Model type: {model_config['model_type']}")
    print(f"{'='*60}\n")

    provider = init_provider(model_config)
    provider.load_model()

    samples = load_samples(lang=lang, split=split, subset=subset, max_samples=max_samples, categories=categories)
    print(f"Loaded {len(samples)} samples from {lang}/{subset}/{split}\n")

    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    results_file = lang_dir / f"{model_key}_{lang}_{subset}_{split}_results.csv"

    results = []
    total_samples = len(samples)
    show_fallback = not sys.stdout.isatty()
    with tqdm(
        total=total_samples,
        desc=f"{model_key}:{lang}:{subset}:{split}",
        unit="it",
        dynamic_ncols=True,
        mininterval=1.0,
        maxinterval=5.0,
        file=sys.stdout,
        leave=True,
    ) as pbar:
        for i, sample in enumerate(samples, 1):
            prompt = format_prompt(sample, prompt_template)
            try:
                response = provider.generate(prompt)
                pred = extract_answer(response)
                answer = str(sample.get("answer", "")).strip().upper()
                is_correct = bool(answer) and (pred == answer)
            except Exception as e:
                response = ""
                pred = ""
                answer = str(sample.get("answer", "")).strip().upper()
                is_correct = False
                print(f"[warn] generation failed on {subset}:{sample.get('id','')} - {e}")

            results.append({
                "id": sample["id"],
                "category": sample["category"],
                "subset": sample["subset"],
                "split": sample["split"],
                "question": sample["question"],
                "A": sample["A"],
                "B": sample["B"],
                "C": sample["C"],
                "D": sample["D"],
                "answer": sample.get("answer", ""),
                "model_response": response,
                "predicted_answer": pred,
                "is_correct": is_correct,
            })

            pbar.update(1)

            if show_fallback and i % 50 == 0:
                print(f"[{i}/{total_samples}] processed")

    pd.DataFrame(results).to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    # Only compute accuracy if answers exist
    has_answers = any(str(r.get("answer", "")).strip() for r in results)
    if has_answers:
        stats = calculate_accuracy(results, by_category=True, by_split=True)
        print_metrics(stats, model_name=model_key)

    provider.unload_model()


def main():
    parser = argparse.ArgumentParser(description="RoleEval bilingual evaluation for DeepSeek-V2-Lite-16B")
    parser.add_argument("--model", default="all", help="Model key or 'all'")
    parser.add_argument("--lang", default="both", choices=["zh", "en", "both"])
    parser.add_argument("--subset", default="both", choices=["chinese", "global", "both"])
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--categories", default=None, help="Comma-separated category names to evaluate")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--config", default=str(Path(__file__).with_name("model_config.yaml")))
    parser.add_argument("--output-dir", default=str(Path(__file__).with_name("results")))
    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    config = load_config(Path(args.config))

    if args.vllm_base_url or args.vllm_model:
        for model_key, model_cfg in config.get("models", {}).items():
            if model_cfg.get("model_type") == "vllm":
                if args.vllm_base_url:
                    model_cfg["base_url"] = args.vllm_base_url
                if args.vllm_model:
                    model_cfg["model_name"] = args.vllm_model

    model_keys = list(config["models"].keys()) if args.model == "all" else [args.model]
    languages = ["zh", "en"] if args.lang == "both" else [args.lang]
    subsets = ["chinese", "global"] if args.subset == "both" else [args.subset]

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    for model_key in model_keys:
        for lang in languages:
            for subset in subsets:
                run_eval(
                    model_key=model_key,
                    config=config,
                    lang=lang,
                    subset=subset,
                    split=args.split,
                    output_dir=Path(args.output_dir),
                    max_samples=args.max_samples,
                    categories=categories,
                )


if __name__ == "__main__":
    main()
