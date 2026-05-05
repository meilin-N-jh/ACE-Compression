#!/usr/bin/env python3
"""CharacterEval response generation using vLLM backend for Llama-70B variants."""
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

# CharacterEval root directory
CHARACTEREVAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHARACTEREVAL_ROOT))

DATA_DIR = CHARACTEREVAL_ROOT / "data"
RESULTS_BASE_DIR = CHARACTEREVAL_ROOT / "llama70B" / "results"


class VLLMClient:
    """vLLM OpenAI-compatible API client (adapted from ToM-Bench)."""

    def __init__(self, base_url: str, model_name: str, timeout: int = 120, max_tokens: int = 512):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens

    def generate(self, messages: list) -> str:
        """Generate response using vLLM API."""
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
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


def concat_messages(conversations, role, system):
    """
    构建对话历史（官方 CharacterEval 方法）
    """
    history = []
    first_query = system
    if conversations[0]['from'] == role:
        first_response = f"好的！现在我来扮演{role}。" + "我首先发话：" + conversations[0]['value']
    else:
        first_response = f"好的！现在我来扮演{role}。"

    history.append({"role": "user", "content": first_query})
    history.append({"role": "assistant", "content": first_response})

    for i in range(len(conversations)):
        if conversations[i]['from'] == role:
            if i == 0:
                continue
            else:
                assert conversations[i-1]['from'] != role
                query = f"{conversations[i-1]['from']}：" + conversations[i-1]['value']
                response = f"{conversations[i]['from']}：" + conversations[i]['value']
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
    assert conversations[-1]['from'] != role

    query = f"{conversations[-1]['from']}：" + conversations[-1]['value']
    return history, query


def make_inputs(context):
    """
    解析上下文为输入格式（官方 CharacterEval 方法）
    """
    dialogues = context.split('\n')
    inputs = []
    for dial in dialogues:
        if '：' in dial:
            role = dial.split("：")[0]
            dial = "：".join(dial.split("：")[1:])
            inputs.append({"from": role, "value": dial})
    return inputs


def get_response_charactereval(data, role_informations, vllm_client):
    """
    生成角色扮演响应（融合官方流程和 vLLM 后端）
    """
    context = data['context']
    role = data['role']

    # 构建角色信息（官方方法）
    role_information = role_informations[role]
    role_system = f'''{role_information}
现在请你扮演一个角色扮演专家。请你根据上述信息扮演{role}进行对话。
'''

    # 构建消息历史（官方方法）
    messages, query = concat_messages(make_inputs(context), role, role_system)

    # 添加当前查询
    messages.append({"role": "user", "content": query})

    # 使用 vLLM API 生成响应
    response = vllm_client.generate(messages)

    data["model_output"] = response
    return data


def main():
    parser = argparse.ArgumentParser(description='CharacterEval evaluation for Llama-70B using vLLM backend')
    parser.add_argument('--model-display-name', type=str, required=True,
                        help='Model display name (e.g., "Llama-70B-FP16")')
    parser.add_argument('--base-url', type=str, required=True,
                        help='vLLM base URL (e.g., "http://127.0.0.1:8002")')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name for vLLM API (e.g., "llama31-70b-fp16")')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file name (default: {model_display_name}_generation.jsonl)')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum tokens to generate (default: 512)')

    args = parser.parse_args()

    # 创建模型专属的 results 目录
    model_results_dir = RESULTS_BASE_DIR / args.model_display_name
    model_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"结果将保存到: {model_results_dir}")

    # 初始化 vLLM 客户端
    print(f"初始化 vLLM 客户端: {args.base_url}")
    print(f"Max tokens: {args.max_tokens}")
    vllm_client = VLLMClient(args.base_url, args.model_name, max_tokens=args.max_tokens)

    # 加载数据（官方方法）
    print("加载测试数据...")
    with open(DATA_DIR / 'test_data.jsonl', 'r') as f:
        datas = json.load(f)

    print("加载角色档案...")
    with open(DATA_DIR / 'character_profiles.json', 'r') as f:
        role_informations = json.load(f)

    print(f"开始生成响应，共 {len(datas)} 个样本")

    results = []
    for idx, data in enumerate(tqdm(datas, desc=f"生成 {args.model_display_name}")):
        result = get_response_charactereval(data, role_informations, vllm_client)
        results.append(result)

        # 每 50 个样本保存一次中间结果
        if (idx + 1) % 50 == 0:
            temp_file = model_results_dir / f"{args.model_display_name}_temp.jsonl"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"\n已保存 {idx+1} 个样本的中间结果")

    # 设置输出文件名
    if args.output_file is None:
        output_file = model_results_dir / f"{args.model_display_name}_generation.jsonl"
    else:
        output_file = model_results_dir / args.output_file

    # 保存最终结果
    print(f"\n保存最终结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("完成！")


if __name__ == '__main__':
    main()
