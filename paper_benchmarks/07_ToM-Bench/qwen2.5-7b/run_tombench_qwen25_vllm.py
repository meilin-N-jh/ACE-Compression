#!/usr/bin/env python3
"""ToM-Bench evaluation for Qwen2.5-7B variants using vLLM backend.
Adapted from official ToM-Bench + qwen2.5-14b implementation.
"""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())

import os
import sys
import argparse
import json
import random
import re
import time
from pathlib import Path
from tqdm import tqdm
import requests
import pandas as pd

# ToM-Bench root directory
TOMBENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOMBENCH_ROOT))

from prompts import *
DATA_DIR = Path(f"{ARTIFACT_ROOT}/paper_benchmarks/07_ToM-Bench/benchmark/data")


def default_results_dir(language: str) -> Path:
    subdir = "results_en" if language == "en" else "results"
    return TOMBENCH_ROOT / "qwen2.5-7b" / subdir


def localized_fields(d: dict, language: str) -> tuple[str, str, dict[str, str]]:
    if language == "en":
        return (
            d.get("STORY", ""),
            d.get("QUESTION", ""),
            {
                "A": d.get("OPTION-A", ""),
                "B": d.get("OPTION-B", ""),
                "C": d.get("OPTION-C", ""),
                "D": d.get("OPTION-D", ""),
            },
        )
    return (
        d.get("故事", ""),
        d.get("问题", ""),
        {
            "A": d.get("选项A", ""),
            "B": d.get("选项B", ""),
            "C": d.get("选项C", ""),
            "D": d.get("选项D", ""),
        },
    )


class VLLMClient:
    """vLLM OpenAI-compatible API client."""

    def __init__(self, base_url: str, model_name: str, timeout: int = 180, seed: int = 42):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.seed = seed
        self.session = requests.Session()
        self.session.trust_env = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "top_p": 1.0,
            "top_k": 0,
            "seed": self.seed,
            "max_tokens": 256,
            "stream": False,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    return ""

                message = choices[0].get("message", {})
                return (message.get("content") or "").strip()

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(1, 3))
                else:
                    print(f"\n[warn] API request failed after {max_retries} attempts: {e}")
                    return ""


def format_prompt_4_choices(d, args):
    """Format prompt for 4-choice questions (handles NaN for 2-choice questions)."""
    if args.language == 'zh':
        cA = d.get('选项A', '')
        cB = d.get('选项B', '')
        cC = d.get('选项C', '')
        cD = d.get('选项D', '')

        try:
            cA = str(cA).replace("A. ", "") if cA and not pd.isna(cA) else ""
            cB = str(cB).replace("B. ", "") if cB and not pd.isna(cB) else ""
            cC = str(cC).replace("C. ", "") if cC and not pd.isna(cC) else ""
            cD = str(cD).replace("D. ", "") if cD and not pd.isna(cD) else ""
        except (TypeError, AttributeError):
            cA = str(cA).replace("A. ", "") if cA else ""
            cB = str(cB).replace("B. ", "") if cB else ""
            cC = str(cC).replace("C. ", "") if cC else ""
            cD = str(cD).replace("D. ", "") if cD else ""

        choices = [c for c in [cA, cB, cC, cD] if c]
        if len(choices) == 2:
            return format_prompt_2_choices(d, args)

        random.shuffle(choices)
        prompt = UserEvaluatePrompt4Choices_zh.format(
            story=d['故事'],
            question=d['问题'],
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3]
        )
        choice_map = {"A": "", "B": "", "C": "", "D": ""}

        for i, choice in enumerate(choices):
            original_label = ""
            if choice == cA:
                original_label = 'A'
            elif choice == cB:
                original_label = 'B'
            elif choice == cC:
                original_label = 'C'
            elif choice == cD:
                original_label = 'D'
            choice_map[chr(65 + i)] = original_label
    else:
        cA = d.get('OPTION-A', '')
        cB = d.get('OPTION-B', '')
        cC = d.get('OPTION-C', '')
        cD = d.get('OPTION-D', '')

        try:
            cA = str(cA).replace("A. ", "") if cA and not pd.isna(cA) else ""
            cB = str(cB).replace("B. ", "") if cB and not pd.isna(cB) else ""
            cC = str(cC).replace("C. ", "") if cC and not pd.isna(cC) else ""
            cD = str(cD).replace("D. ", "") if cD and not pd.isna(cD) else ""
        except (TypeError, AttributeError):
            cA = str(cA).replace("A. ", "") if cA else ""
            cB = str(cB).replace("B. ", "") if cB else ""
            cC = str(cC).replace("C. ", "") if cC else ""
            cD = str(cD).replace("D. ", "") if cD else ""

        choices = [c for c in [cA, cB, cC, cD] if c]
        if len(choices) == 2:
            return format_prompt_2_choices(d, args)

        random.shuffle(choices)
        prompt = UserEvaluatePrompt4Choices_en.format(
            story=d['STORY'],
            question=d['QUESTION'],
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3]
        )
        choice_map = {"A": "", "B": "", "C": "", "D": ""}

        for i, choice in enumerate(choices):
            original_label = ""
            if choice == cA:
                original_label = 'A'
            elif choice == cB:
                original_label = 'B'
            elif choice == cC:
                original_label = 'C'
            elif choice == cD:
                original_label = 'D'
            choice_map[chr(65 + i)] = original_label

    return choice_map, prompt


def format_prompt_2_choices(d, args):
    """Format prompt for 2-choice questions."""
    if args.language == 'zh':
        cA = d['选项A'].replace("A. ", "")
        cB = d['选项B'].replace("B. ", "")
        choices = [cA, cB]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt2Choices_zh.format(
            story=d['故事'],
            question=d['问题'],
            choice_a=choices[0],
            choice_b=choices[1]
        )
        choice_map = {"A": "", "B": "", "C": "", "D": ""}
        for i, choice in enumerate(choices):
            original_label = ""
            if choice == cA:
                original_label = 'A'
            elif choice == cB:
                original_label = 'B'
            choice_map[chr(65 + i)] = original_label
    else:
        cA = d['OPTION-A'].replace("A. ", "")
        cB = d['OPTION-B'].replace("B. ", "")
        choices = [cA, cB]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt2Choices_en.format(
            story=d['STORY'],
            question=d['QUESTION'],
            choice_a=choices[0],
            choice_b=choices[1]
        )
        choice_map = {"A": "", "B": "", "C": "", "D": ""}
        for i, choice in enumerate(choices):
            original_label = ""
            if choice == cA:
                original_label = 'A'
            elif choice == cB:
                original_label = 'B'
            choice_map[chr(65 + i)] = original_label

    return choice_map, prompt


def parse_answer(text):
    if "[[A]]" in text:
        return "A"
    if "[[B]]" in text:
        return "B"
    if "[[C]]" in text:
        return "C"
    if "[[D]]" in text:
        return "D"
    if "[A]" in text:
        return "A"
    if "[B]" in text:
        return "B"
    if "[C]" in text:
        return "C"
    if "[D]" in text:
        return "D"
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ['A', 'B', 'C', 'D']:
            return text[i]
    return "A"


def run_task(task, args, vllm_client):
    file_path = DATA_DIR / f"{task}.jsonl"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f.readlines()]
    if args.max_samples is not None:
        data = data[:args.max_samples]

    results = []
    for q_idx, d in enumerate(tqdm(data, desc=f"{task}")):
        if args.language == 'zh':
            system_prompt = SystemEvaluatePrompt_zh
        else:
            system_prompt = SystemEvaluatePrompt_en

        if d.get('选项C', None) is None or (isinstance(d.get('选项C', None), float) and pd.isna(d.get('选项C'))):
            choice_map, user_prompt = format_prompt_2_choices(d, args)
        else:
            choice_map, user_prompt = format_prompt_4_choices(d, args)

        for t in range(args.try_times):
            output = vllm_client.generate(system_prompt, user_prompt)
            pred = parse_answer(output)
            mapped_pred = choice_map.get(pred, "")
            story, question, choices = localized_fields(d, args.language)
            answer = d.get('答案\nANSWER', d.get('答案', d.get('ANSWER', '')))
            results.append({
                "question_idx": q_idx,
                "trial": t,
                "language": args.language,
                "story": story,
                "question": question,
                "choices": choices,
                "raw_prediction": pred,
                "mapped_prediction": mapped_pred,
                "model_response": output,
                "answer": answer,
                "is_correct": mapped_pred == answer if mapped_pred else False,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-display-name", type=str, required=True)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--try-times", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    results_dir = Path(os.environ.get("TOMBENCH_RESULTS_DIR", str(default_results_dir(args.language))))
    results_dir.mkdir(parents=True, exist_ok=True)

    vllm_client = VLLMClient(args.base_url, args.model_name, seed=args.seed)

    tasks = []
    with open(TOMBENCH_ROOT / "all_tasks.txt", "r", encoding='utf-8') as f:
        for line in f:
            # Remove 'data/' prefix and '.jsonl' suffix
            task_name = line.strip()
            if task_name.startswith('data/'):
                task_name = task_name[5:]
            if task_name.endswith('.jsonl'):
                task_name = task_name[:-6]
            tasks.append(task_name)

    for task in tasks:
        output_file = results_dir / f"{task}_{args.model_display_name}_results.jsonl"
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"[skip] Existing task results found for {task}: {output_file.name}")
            continue
        task_results = run_task(task, args, vllm_client)
        with open(output_file, "w", encoding='utf-8') as f:
            for record in task_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"All tasks completed. Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
