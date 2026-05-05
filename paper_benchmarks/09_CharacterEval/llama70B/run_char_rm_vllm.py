#!/usr/bin/env python3
"""Run CharacterRM evaluation (adapted from official for llama70B)."""
import sys
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm

# CharacterEval root directory
CHARACTEREVAL_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CHARACTEREVAL_ROOT / "data"

# 导入 BaichuanCharRM
sys.path.insert(0, str(CHARACTEREVAL_ROOT))
from BaichuanCharRM.modeling_baichuan import BaichuanCharRM
from BaichuanCharRM.tokenization_baichuan import BaichuanTokenizer

MAX_SEQ_LENGTH = 4096


def format_input(example, character_profile):
    """格式化输入（官方方法）"""
    input_text = "<RoleInfo>\n\n" \
        + str(character_profile[example['role']]) + "\n\n<Context>\n\n" \
        + example['context'] + "\n\n<Response>\n\n" \
        + example['model_output'] + "\n\n<Dimension>\n\n" \
        + example["metric_zh"]
    return input_text


def main():
    parser = argparse.ArgumentParser(description='Run CharacterRM evaluation')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name (e.g., "llama31-70b-4bit")')
    parser.add_argument('--display-name', type=str, default=None,
                        help='Display name for results directory (e.g., "Llama-70B-4bit")')
    parser.add_argument('--reward-model-path', type=str,
                        default='BaichuanCharRM/',
                        help='Path to BaichuanCharRM model')

    args = parser.parse_args()

    # Use display_name if provided, otherwise derive from model_name
    display_name = args.display_name if args.display_name else args.model_name.replace("-", " ").title().replace(" ", "-")

    # 构建路径
    model_results_dir = CHARACTEREVAL_ROOT / "llama70B" / "results" / display_name
    input_file = model_results_dir / f"{display_name}_generation_trans.jsonl"
    output_file = model_results_dir / f"{display_name}_evaluation.jsonl"

    # 加载角色档案（官方方法）
    print("加载角色档案...")
    with open(DATA_DIR / 'character_profiles.json', "r", encoding='utf-8') as f:
        character_profile = json.load(f)

    # 加载转换后的数据
    print(f"加载转换后的数据: {input_file}")
    with open(input_file, mode='r', encoding='utf-8') as f:
        records = json.load(f)

    print(f"共 {len(records)} 条评测记录")

    # 加载 CharacterRM 模型（官方方法）
    print(f"加载 CharacterRM 模型: {args.reward_model_path}")
    tokenizer = BaichuanTokenizer.from_pretrained(args.reward_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = BaichuanCharRM.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.bfloat16
    ).cuda()

    # 运行评测
    print("开始 CharacterRM 评分...")
    for record in tqdm(records, desc=f"评分 {args.model_name}"):
        input_text = format_input(record, character_profile)
        input_ids = tokenizer.encode(
            text=input_text,
            add_special_tokens=False
        ) + [tokenizer.eos_token_id]

        # 截断到最大长度
        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[-MAX_SEQ_LENGTH:]

        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

        with torch.no_grad():
            score = base_model(input_ids=input_ids)[1].item() * 4 + 1
            record[record['metric_en']] = score

    # 保存评测结果
    print(f"\n保存评测结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)

    print("CharacterRM 评分完成！")


if __name__ == '__main__':
    main()
