#!/usr/bin/env python3
"""
HumanEval 模型评测脚本
使用transformers库，支持官方chat template和量化版本
支持全精度、8bit、4bit量化
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
import json
import re
import argparse
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用GPU2

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("错误: 请安装transformers库")
    print("安装命令: pip install transformers torch")
    sys.exit(1)

class HumanEvalEvaluator:
    def __init__(self, model_path, device="auto", torch_dtype="float16", quantization="none"):
        """初始化评测器"""
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch.float16 if torch_dtype == "float16" else torch.float32
        self.quantization = quantization

        print(f"正在加载模型: {model_path}")
        print(f"设备: {device}")
        print(f"数据类型: {torch_dtype}")
        print(f"量化类型: {quantization}")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 根据量化类型加载模型
            if quantization == "8bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config
                )
                print(f"✓ 8bit量化模型加载成功")

            elif quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    llm_int8_threshold=6.0
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    quantization_config=quantization_config
                )
                print(f"✓ 4bit量化模型加载成功")

            else:  # none (full precision)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.torch_dtype,
                    device_map=device,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print(f"✓ 全精度模型加载成功")

            print(f"模型参数量: {sum(p.numel() for p in self.model.parameters())/1e9:.1f}B")

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise

    def clean_code_output(self, raw_output):
        """
        简化的代码清理，主要处理markdown标记
        """
        if not raw_output:
            return "    pass  # Empty output"

        output = raw_output.strip()

        # 提取markdown代码块中的Python代码
        python_block_pattern = r'```python\s*\n(.*?)\n```'
        matches = re.findall(python_block_pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # 移除markdown标记
        output = re.sub(r'```python\s*\n', '', output, flags=re.IGNORECASE)
        output = re.sub(r'```', '', output)

        # 移除常见的前缀文本
        prefixes = [
            "Here's the completed function:",
            "The completed function is:",
            "Here is the implementation:",
            "Code:",
        ]

        for prefix in prefixes:
            if output.startswith(prefix):
                output = output[len(prefix):].strip()
                break

        return output if output else "    pass  # Cleaned output was empty"

    def create_chat_prompt(self, problem_prompt):
        """
        使用官方chat template创建prompt
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. When completing Python functions, respond ONLY with the code itself. Do not use markdown formatting like ```python``` blocks. Do not include explanations. Just provide the pure Python code starting with def or necessary imports."
            },
            {
                "role": "user",
                "content": f"Complete this Python function:\n\n{problem_prompt}\n\nProvide ONLY the completed function code without any explanations or markdown formatting."
            }
        ]

        # 使用官方chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: Chat template failed, using fallback: {e}")
            # fallback格式
            prompt = f"""<|im_start|>system
You are a helpful assistant. When completing Python functions, respond ONLY with the code itself. Do not use markdown formatting like ```python``` blocks. Do not include explanations. Just provide the pure Python code starting with def or necessary imports.<|im_end|>
<|im_start|>user
Complete this Python function:

{problem_prompt}

Provide ONLY the completed function code without any explanations or markdown formatting.<|im_end|>
<|im_start|>assistant
"""

        return prompt

    def generate_completion(self, prompt, temperature=0.1, max_tokens=256):
        """
        生成代码补全
        """
        try:
            # 编码输入
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )

            # 移动到正确的设备
            if self.device == "auto":
                device = next(self.model.parameters()).device
            else:
                device = self.device

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 生成响应
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=False if temperature <= 0.01 else True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            # 解码响应，只取新生成的部分
            raw_response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # 后处理：清理代码输出
            cleaned_response = self.clean_code_output(raw_response)

            return cleaned_response

        except Exception as e:
            print(f"生成补全时出错: {e}")
            return "    pass"

    def evaluate_problems(self, num_samples_per_task=1, temperature=0.1):
        """
        评测所有HumanEval问题
        """
        print("读取HumanEval问题集...")
        problems = read_problems()
        print(f"共找到 {len(problems)} 个问题")

        samples = []

        # 使用tqdm显示进度
        with tqdm(total=len(problems) * num_samples_per_task, desc="生成代码补全") as pbar:
            for task_id, problem in problems.items():
                for _ in range(num_samples_per_task):
                    # 创建chat prompt
                    prompt = self.create_chat_prompt(problem['prompt'])

                    # 生成补全
                    completion = self.generate_completion(
                        prompt,
                        temperature=temperature
                    )

                    # 保存样本
                    samples.append({
                        'task_id': task_id,
                        'completion': completion
                    })

                    pbar.update(1)

                    # 可选：打印示例进度和清理效果
                    if len(samples) % 10 == 0:
                        print(f"\n已完成 {len(samples)} 个样本...")
                        print(f"最新样本 task_id: {task_id}")
                        print(f"清理后代码: {completion[:150]}...")

                        # 每50个样本显示一个完整的示例对比
                        if len(samples) % 50 == 0:
                            print(f"\n完整示例 (task_id: {task_id}):")
                            print(f"代码:\n{completion}")
                            print("-" * 50)

        return samples

def main():
    parser = argparse.ArgumentParser(description="HumanEval 模型评测")

    # 模型配置
    parser.add_argument("--model-path",
                       default=f"{ARTIFACT_ROOT}/models/Qwen-7B-Chat",
                       help="模型路径")
    parser.add_argument("--quantization",
                       choices=["none", "8bit", "4bit"],
                       default="none",
                       help="量化类型")

    # 评测配置
    parser.add_argument("--samples", type=int, default=1,
                       help="每个问题的样本数量")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="生成温度")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="最大生成token数")
    parser.add_argument("--device", default="auto",
                       help="设备配置")
    parser.add_argument("--torch-dtype", default="float16",
                       choices=["float16", "float32"],
                       help="数据类型")

    # 输出配置
    parser.add_argument("--output",
                       default=None,
                       help="输出文件名（默认根据量化类型生成）")

    args = parser.parse_args()

    # 根据量化类型设置默认输出文件名
    if args.output is None:
        if args.quantization == "none":
            args.output = "qwen_full_precision_human_eval.jsonl"
        elif args.quantization == "8bit":
            args.output = "qwen_8bit_human_eval.jsonl"
        elif args.quantization == "4bit":
            args.output = "qwen_4bit_human_eval.jsonl"

    quantization_name = {
        "none": "全精度",
        "8bit": "8bit量化",
        "4bit": "4bit量化"
    }

    print("=" * 60)
    print(f"HumanEval {quantization_name[args.quantization]}模型评测")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"量化类型: {args.quantization}")
    print(f"样本数量/问题: {args.samples}")
    print(f"温度: {args.temperature}")
    print(f"设备: {args.device}")
    print(f"数据类型: {args.torch_dtype}")
    print(f"输出文件: {args.output}")
    print("=" * 60)

    # 初始化评测器
    try:
        evaluator = HumanEvalEvaluator(
            model_path=args.model_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization
        )
    except Exception as e:
        print(f"评测器初始化失败: {e}")
        return 1

    # 运行评测
    print(f"\n开始生成 {args.samples} 个样本/问题...")
    samples = evaluator.evaluate_problems(
        num_samples_per_task=args.samples,
        temperature=args.temperature
    )

    # 保存结果
    print(f"\n保存结果到 {args.output}...")
    write_jsonl(args.output, samples)

    print(f"\n✓ 评测完成!")
    print(f"共生成 {len(samples)} 个样本")
    print(f"输出文件: {args.output}")

    # 运行评测命令提示
    print(f"\n运行功能正确性评测:")
    print(f"python3 -c \"from human_eval.evaluate_functional_correctness import evaluate_functional_correctness; evaluate_functional_correctness('{args.output}')\"")

    return 0

if __name__ == "__main__":
    exit(main())