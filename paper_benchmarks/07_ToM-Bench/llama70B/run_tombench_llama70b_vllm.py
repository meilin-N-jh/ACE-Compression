#!/usr/bin/env python3
"""ToM-Bench evaluation for Llama-70B variants using vLLM backend."""
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
import pandas as pd  # For handling NaN values

# ToM-Bench root directory
TOMBENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOMBENCH_ROOT))

from prompts import *
DATA_DIR = TOMBENCH_ROOT / "data"


def default_results_dir(language: str) -> Path:
    subdir = "results_en" if language == "en" else "results"
    return TOMBENCH_ROOT / "llama70B" / subdir


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

    def __init__(self, base_url: str, model_name: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.session = requests.Session()
        self.session.trust_env = False

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using vLLM API."""
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
        # Handle NaN values - convert to empty string
        cA = d.get('选项A', '')
        cB = d.get('选项B', '')
        cC = d.get('选项C', '')
        cD = d.get('选项D', '')

        # Check if any option is NaN or missing
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

        # Determine actual number of choices
        choices = [c for c in [cA, cB, cC, cD] if c]

        # If only 2 choices, use 2-choice format
        if len(choices) == 2:
            return format_prompt_2_choices(d, args)

        # Otherwise format as 4-choice question
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

        # Map shuffled choices back to original labels
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
            choice_map[chr(65 + i)] = original_label  # 65 is 'A' in ASCII
    else:
        # English version - handle NaN values
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

        # If only 2 choices, use 2-choice format
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


def load_task_data(task_file: str) -> list:
    """Load task data from jsonl file."""
    with open(task_file, "r", encoding='utf-8') as f:
        return [json.loads(line) for line in f.readlines()]


def evaluate_task(
    client: VLLMClient,
    task_name: str,
    data: list,
    args,
    output_file: Path
):
    """Evaluate a single task."""

    # Set system prompt based on language and CoT
    if args.language == "zh":
        system_prompt = SystemEvaluatePrompt_zh_cot if args.cot else SystemEvaluatePrompt_zh
    else:
        system_prompt = SystemEvaluatePrompt_en_cot if args.cot else SystemEvaluatePrompt_en

    results = []
    total_questions = len(data) * args.try_times

    print(f"\nTask: {task_name}")
    print(f"Questions: {len(data)} × {args.try_times} trials = {total_questions} evaluations")

    with tqdm(total=total_questions, desc=f"{task_name}", unit="eval") as pbar:
        for question_idx, d in enumerate(data):
            # Determine if this is a 2-choice or 4-choice question
            has_option_c = d.get('选项C', None) is not None

            for trial in range(args.try_times):
                # Format prompt
                if has_option_c:
                    choice_map, user_prompt = format_prompt_4_choices(d, args)
                else:
                    choice_map, user_prompt = format_prompt_2_choices(d, args)

                # Get model response
                response = client.generate(system_prompt, user_prompt)

                # Extract prediction - following official ToM-Bench format
                # Priority: [[A]] > [A] > single character (from beginning)
                pred = ""
                response_upper = response.strip().upper()

                # Priority 1: [[A]] format (FIRST occurrence, following official)
                if "[[A]]" in response_upper:
                    pred = "A"
                elif "[[B]]" in response_upper:
                    pred = "B"
                elif "[[C]]" in response_upper:
                    pred = "C"
                elif "[[D]]" in response_upper:
                    pred = "D"
                # Priority 2: [A] format (FIRST occurrence)
                elif "[A]" in response_upper:
                    pred = "A"
                elif "[B]" in response_upper:
                    pred = "B"
                elif "[C]" in response_upper:
                    pred = "C"
                elif "[D]" in response_upper:
                    pred = "D"
                # Priority 3: single character from end (following official)
                else:
                    for i in range(len(response_upper) - 1, -1, -1):
                        if response_upper[i] == 'A':
                            pred = "A"
                            break
                        elif response_upper[i] == 'B':
                            pred = "B"
                            break
                        elif response_upper[i] == 'C':
                            pred = "C"
                            break
                        elif response_upper[i] == 'D':
                            pred = "D"
                            break
                    # Default fallback (following official)
                    if not pred:
                        pred = "A"

                # Map prediction back to original choice label
                if pred and pred in choice_map:
                    mapped_pred = choice_map[pred]
                else:
                    mapped_pred = ""

                # Get ground truth answer
                answer = d['答案\nANSWER'].strip().upper()

                story, question, choices = localized_fields(d, args.language)
                result = {
                    "question_idx": question_idx,
                    "trial": trial,
                    "language": args.language,
                    "story": story,
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "model_response": response,
                    "predicted_choice": pred,
                    "mapped_prediction": mapped_pred,
                    "is_correct": mapped_pred == answer if mapped_pred else False,
                }
                results.append(result)
                pbar.update(1)

    # Save results to jsonl
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Calculate accuracy
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    accuracy = correct / total * 100 if total > 0 else 0

    print(f"Accuracy: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Run ToM-Bench evaluation with Llama-70B models via vLLM")
    parser.add_argument("--model-display-name", type=str, required=True, help="Display name for results (e.g., Llama-70B-FP16)")
    parser.add_argument("--base-url", type=str, required=True, help="vLLM API base URL (e.g., http://127.0.0.1:8001)")
    parser.add_argument("--model-name", type=str, required=True, help="Model name in vLLM (e.g., llama31-70b-fp16)")
    parser.add_argument("--language", type=str, default="zh", choices=["zh", "en"], help="Language (zh/en)")
    parser.add_argument("--try-times", type=int, default=5, help="Number of trials per question")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--task", type=str, default="", help="Specific task to run (empty for all tasks)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout", type=int, default=120, help="API request timeout in seconds")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    results_dir = Path(os.environ.get("TOMBENCH_RESULTS_DIR", str(default_results_dir(args.language))))

    # Initialize vLLM client
    client = VLLMClient(args.base_url, args.model_name, args.timeout)

    print(f"\n{'='*70}")
    print(f"ToM-Bench Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model_display_name}")
    print(f"Base URL: {args.base_url}")
    print(f"Model Name: {args.model_name}")
    print(f"Language: {args.language}")
    print(f"Trials per question: {args.try_times}")
    print(f"Chain-of-Thought: {args.cot}")
    print(f"{'='*70}\n")

    # Get task files
    task_files = sorted(DATA_DIR.glob("*.jsonl"))
    if args.task:
        wanted_tasks = args.task.split(',')
        task_files = [f for f in task_files if f.name in wanted_tasks]

    print(f"Found {len(task_files)} task files to evaluate\n")

    # Evaluate each task
    all_results = {}
    for task_file in task_files:
        task_name = task_file.stem

        try:
            output_file = results_dir / f"{task_name}_{args.model_display_name}_results.jsonl"
            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"[skip] Existing task results found for {task_name}: {output_file.name}")
                continue
            data = load_task_data(task_file)
            accuracy = evaluate_task(client, task_name, data, args, output_file)
            all_results[task_name] = {
                "accuracy": accuracy,
                "correct": sum(1 for r in all_results.values()),
                "total": len(data) * args.try_times,
            }
        except Exception as e:
            print(f"\n[error] Failed to evaluate task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for task_name, result in all_results.items():
        print(f"{task_name}: {result['accuracy']:.2f}%")

    if all_results:
        avg_accuracy = sum(r['accuracy'] for r in all_results.values()) / len(all_results)
        print(f"\nAverage Accuracy: {avg_accuracy:.2f}%")

    print(f"\nResults saved to: {results_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
