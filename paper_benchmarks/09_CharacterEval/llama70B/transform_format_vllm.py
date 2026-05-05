#!/usr/bin/env python3
"""Transform format for CharacterRM evaluation (adapted from official)."""
import sys
import argparse
import json
import copy
from pathlib import Path

# CharacterEval root directory
CHARACTEREVAL_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CHARACTEREVAL_ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description='Transform generation format for CharacterRM')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input generation file (e.g., "llama70B/results/Llama-70B-FP16_generation.jsonl")')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output transformed file (e.g., "llama70B/results/Llama-70B-FP16_generation_trans.jsonl")')

    args = parser.parse_args()

    # 加载 id2metric 映射（官方方法）
    print("加载 id2metric 映射...")
    with open(DATA_DIR / 'id2metric.jsonl', 'r') as f:
        id_metric = json.load(f)

    # 加载生成结果
    print(f"加载生成结果: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)

    print(f"开始转换格式，共 {len(datas)} 个样本")

    results = []
    for data in datas:
        # 过滤无效输出（官方方法）
        if data['model_output'] is not None and data['model_output'] != "ERROR":
            # 只保留第一行，防止连续生成
            model_output = data['model_output'].split("\n")[0]
            data['model_output'] = model_output

            # 根据 id 添加对应的评测维度
            if str(data['id']) in id_metric:
                for x in id_metric[str(data['id'])]:
                    data['metric_en'] = x[0]
                    data['metric_zh'] = x[1]
                    tmp = copy.deepcopy(data)
                    results.append(tmp)

    # 保存转换后的结果
    print(f"保存转换结果到: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"转换完成！共生成 {len(results)} 条评测记录")


if __name__ == '__main__':
    main()
