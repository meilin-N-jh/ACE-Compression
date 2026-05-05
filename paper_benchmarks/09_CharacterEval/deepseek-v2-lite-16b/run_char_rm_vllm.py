#!/usr/bin/env python3
"""Run CharacterRM evaluation for DeepSeek-V2-Lite-16B."""
import sys
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

CHARACTEREVAL_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CHARACTEREVAL_ROOT / "data"

sys.path.insert(0, str(CHARACTEREVAL_ROOT))
from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer

MAX_SEQ_LENGTH = 4096


def format_input(example, character_profile):
    input_text = "<RoleInfo>\n\n" \
        + str(character_profile[example['role']]) + "\n\n<Context>\n\n" \
        + example['context'] + "\n\n<Response>\n\n" \
        + example['model_output'] + "\n\n<Dimension>\n\n" \
        + example["metric_zh"]
    return input_text


def main():
    parser = argparse.ArgumentParser(description='Run CharacterRM evaluation for DeepSeek-V2-Lite-16B')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name (e.g., "deepseek-v2-lite-fp16")')
    parser.add_argument('--display-name', type=str, default=None,
                        help='Display name for results directory (e.g., "DeepSeek-V2-Lite-16B-FP16")')
    parser.add_argument('--reward-model-path', type=str,
                        default=str(CHARACTEREVAL_ROOT / "BaichuanCharRM/"),
                        help='Path to BaichuanCharRM model')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU to use (-1 for auto-detect via CUDA_VISIBLE_DEVICES)')

    args = parser.parse_args()

    # Use display_name if provided, otherwise derive from model_name
    display_name = args.display_name if args.display_name else args.model_name.replace("-", " ").title().replace(" ", "-")

    model_results_dir = Path(__file__).resolve().parent / "results" / display_name
    input_file = model_results_dir / f"{display_name}_generation_trans.jsonl"
    output_file = model_results_dir / f"{display_name}_evaluation.jsonl"

    print("加载角色档案...")
    with open(DATA_DIR / 'character_profiles.json', "r", encoding='utf-8') as f:
        character_profile = json.load(f)

    print(f"加载转换后的数据: {input_file}")
    with open(input_file, mode='r', encoding='utf-8') as f:
        records = json.load(f)

    print(f"共 {len(records)} 条评测记录")

    # Use cuda:0, controlled by CUDA_VISIBLE_DEVICES environment variable
    if args.gpu == -1:
        import os
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        print(f"使用 GPU 0 (通过 CUDA_VISIBLE_DEVICES={cuda_visible} 控制)")
    args.gpu = 0  # Always use cuda:0

    print(f"加载 CharacterRM 模型: {args.reward_model_path}")
    tokenizer = BaichuanTokenizer.from_pretrained(args.reward_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = BaichuanCharRM.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.bfloat16
    ).cuda(args.gpu)

    print("开始 CharacterRM 评分...")
    for record in tqdm(records, desc=f"评分 {args.model_name}"):
        input_text = format_input(record, character_profile)
        input_ids = tokenizer.encode(
            text=input_text,
            add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[-MAX_SEQ_LENGTH:]

        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda(args.gpu)

        with torch.no_grad():
            score = base_model(input_ids=input_ids)[1].item() * 4 + 1
            record[record['metric_en']] = score

    print(f"\n保存评测结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("完成!")


if __name__ == "__main__":
    main()
