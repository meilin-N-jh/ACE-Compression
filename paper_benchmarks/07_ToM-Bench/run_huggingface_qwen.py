import json
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from prompts import *
from tqdm import tqdm
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def format_prompt_4(d, args):
    if args.language == 'zh':
        cA = d['选项A'].replace("A. ", "")
        cB = d['选项B'].replace("B. ", "")
        cC = d['选项C'].replace("C. ", "")
        cD = d['选项D'].replace("D. ", "")
        choices = [cA, cB, cC, cD]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt4Choices_zh.format(story=d['故事'], question=d['问题'], choice_a=choices[0], choice_b=choices[1], choice_c=choices[2], choice_d=choices[3])
        map = {"A": "", "B": "", "C": "", "D": ""}

        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'
        elif choices[0] == cC:
            map['A'] = 'C'
        elif choices[0] == cD:
            map['A'] = 'D'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'
        elif choices[1] == cC:
            map['B'] = 'C'
        elif choices[1] == cD:
            map['B'] = 'D'

        if choices[2] == cA:
            map['C'] = 'A'
        elif choices[2] == cB:
            map['C'] = 'B'
        elif choices[2] == cC:
            map['C'] = 'C'
        elif choices[2] == cD:
            map['C'] = 'D'

        if choices[3] == cA:
            map['D'] = 'A'
        elif choices[3] == cB:
            map['D'] = 'B'
        elif choices[3] == cC:
            map['D'] = 'C'
        elif choices[3] == cD:
            map['D'] = 'D'
    else:
        cA = d['OPTION-A'].replace("A. ", "")
        cB = d['OPTION-B'].replace("B. ", "")
        cC = d['OPTION-C'].replace("C. ", "")
        cD = d['OPTION-D'].replace("D. ", "")
        choices = [cA, cB, cC, cD]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt4Choices_en.format(story=d['STORY'], question=d['QUESTION'], choice_a=choices[0], choice_b=choices[1], choice_c=choices[2], choice_d=choices[3])
        map = {"A": "", "B": "", "C": "", "D": ""}

        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'
        elif choices[0] == cC:
            map['A'] = 'C'
        elif choices[0] == cD:
            map['A'] = 'D'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'
        elif choices[1] == cC:
            map['B'] = 'C'
        elif choices[1] == cD:
            map['B'] = 'D'

        if choices[2] == cA:
            map['C'] = 'A'
        elif choices[2] == cB:
            map['C'] = 'B'
        elif choices[2] == cC:
            map['C'] = 'C'
        elif choices[2] == cD:
            map['C'] = 'D'

        if choices[3] == cA:
            map['D'] = 'A'
        elif choices[3] == cB:
            map['D'] = 'B'
        elif choices[3] == cC:
            map['D'] = 'C'
        elif choices[3] == cD:
            map['D'] = 'D'
    return map, prompt


def format_prompt_2(d, args):
    if args.language == 'zh':
        cA = d['选项A'].replace("A. ", "")
        cB = d['选项B'].replace("B. ", "")
        choices = [cA, cB]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt2Choices_zh.format(story=d['故事'], question=d['问题'], choice_a=choices[0], choice_b=choices[1])
        map = {"A": "", "B": "", "C": "", "D": ""}
        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'
    else:
        cA = d['OPTION-A'].replace("A. ", "")
        cB = d['OPTION-B'].replace("B. ", "")
        choices = [cA, cB]
        random.shuffle(choices)
        prompt = UserEvaluatePrompt2Choices_en.format(story=d['STORY'], question=d['QUESTION'], choice_a=choices[0], choice_b=choices[1])
        map = {"A": "", "B": "", "C": "", "D": ""}
        if choices[0] == cA:
            map['A'] = 'A'
        elif choices[0] == cB:
            map['A'] = 'B'

        if choices[1] == cA:
            map['B'] = 'A'
        elif choices[1] == cB:
            map['B'] = 'B'

    return map, prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--try_times", type=int, default=5)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (for testing)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with specific settings for Qwen
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model_name = args.model_name.split("/")[-1]

    # Ensure results directory exists
    os.makedirs("./results", exist_ok=True)

    files = os.listdir("./data")
    if args.task != "":
        files = [args.task] if not args.task.endswith('.jsonl') else [args.task]

    for file in files:
        if not file.endswith('.jsonl'):
            continue

        task = file.split(".")[0]
        print(f"Processing task: {task}")

        with open(f"data/{file}", "r", encoding='utf-8') as f:
            data = [json.loads(line) for line in f.readlines()]

        # Limit samples for testing if specified
        if args.max_samples:
            data = data[:args.max_samples]
            print(f"Limited to {len(data)} samples")

        for i, d in tqdm(enumerate(data), desc=f"Processing {task}"):
            for j in range(args.try_times):
                try:
                    # Check if there are 4 choices or 2 choices (official method)
                    if d['选项C'] != None:
                        maps, prompt = format_prompt_4(d, args)
                    else:
                        maps, prompt = format_prompt_2(d, args)

                    system_prompt = ""
                    if args.language == "zh":
                        if args.cot == False:
                            system_prompt = SystemEvaluatePrompt_zh
                        else:
                            system_prompt = SystemEvaluatePrompt_zh_cot
                    else:
                        if args.cot == False:
                            system_prompt = SystemEvaluatePrompt_en
                        else:
                            system_prompt = SystemEvaluatePrompt_en_cot

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]

                    # Create the prompt text for Qwen (simpler approach)
                    full_prompt = system_prompt + "\n\n" + prompt

                    # Tokenize and generate (Qwen-specific approach)
                    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True)

                    # Move to device
                    if hasattr(model, 'device'):
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    else:
                        inputs = {k: v.to('cuda') for k, v in inputs.items() if torch.is_tensor(v)}

                    # Generate with Qwen-compatible parameters
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs.get('attention_mask'),
                            max_new_tokens=512,
                            do_sample=False,
                            temperature=0.0,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )

                    # Decode only the generated part
                    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

                    out = {}
                    out['idx'] = i
                    out['number'] = j
                    out['answer'] = d['答案\nANSWER']
                    out['map'] = maps
                    out['data'] = d
                    out['output'] = response

                    # Add FP16 suffix to model name for result files
                    result_model_name = f"{model_name}_fp16"

                    # 创建模型命名的子文件夹
                    model_dir = f"./results/{model_name}"
                    os.makedirs(model_dir, exist_ok=True)

                    with open(f"{model_dir}/{task}_{result_model_name}_results.jsonl", "a+", encoding='utf-8') as f:
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")

                except Exception as e:
                    print(f"Error processing sample {i}, attempt {j}: {str(e)}")
                    continue

    print(f"Evaluation completed for {model_name}!")