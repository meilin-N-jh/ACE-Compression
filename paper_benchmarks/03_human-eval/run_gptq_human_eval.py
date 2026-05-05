#!/usr/bin/env python3
"""
HumanEval GPTQ量化模型评测脚本
参考PsychoBench的GPTQ模型加载方式
支持官方chat template和后处理清理
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
import time
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用GPU2

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("错误: 请安装transformers库")
    print("安装命令: pip install transformers torch")
    sys.exit(1)

class GPTQHumanEvalEvaluator:
    def __init__(self, model_path, n_ctx=4096, seed=42):
        """初始化GPTQ评测器"""
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.seed = seed
        torch.manual_seed(seed)

        print(f"正在加载GPTQ模型: {model_path}")
        print(f"上下文长度: {n_ctx}")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 使用PsychoBench成功的模型加载方式
            print("使用PsychoBench成功的GPTQ加载方式...")
            model_start_time = time.time()

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model.eval()

            model_load_time = time.time() - model_start_time
            print(f"✓ GPTQ模型加载成功")
            print(f"✓ 模型类型: {type(self.model)}")
            print(f"✓ 加载时间: {model_load_time:.1f}秒")

            device = next(self.model.parameters()).device
            print(f"✓ 模型设备: {device}")

            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / (1024**3)
                print(f"✓ 显存使用: {memory_gb:.1f} GB")

        except Exception as e:
            print(f"✗ GPTQ模型加载失败: {e}")
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
        创建chat prompt用于代码生成 - 优化中文prompt
        """
        # 使用更简洁直接的中文prompt，符合Qwen模型的训练风格
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的Python程序员。请直接返回完整的函数代码，不要使用markdown代码块格式，不要添加任何解释文字，只需要代码。"
            },
            {
                "role": "user",
                "content": f"请补全以下Python函数：\n\n{problem_prompt}\n\n直接返回函数代码，无需解释。"
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
            # fallback格式 - 直接使用Qwen的chat格式
            prompt = f"""<|im_start|>system
你是一个专业的Python程序员。请直接返回完整的函数代码，不要使用markdown代码块格式，不要添加任何解释文字，只需要代码。<|im_end|>
<|im_start|>user
请补全以下Python函数：

{problem_prompt}

直接返回函数代码，无需解释。<|im_end|>
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
                max_length=self.n_ctx
            )

            # 确保attention_mask存在
            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            # 移动到正确的设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

            # 生成响应 - 使用PsychoBench成功的参数
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_tokens,
                    do_sample=True,  # GPTQ模型建议使用采样
                    temperature=0.01 if temperature <= 0.01 else temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                    return_dict_in_generate=True
                )

            # 解码响应，只取新生成的部分
            generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
            raw_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
    parser = argparse.ArgumentParser(description="HumanEval GPTQ量化模型评测")

    # 模型配置
    parser.add_argument("--model-path",
                       default=f"{ARTIFACT_ROOT}/models/Qwen-7B-Chat-GPTQ",
                       help="GPTQ模型路径")

    # 评测配置
    parser.add_argument("--samples", type=int, default=1,
                       help="每个问题的样本数量")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="生成温度")
    parser.add_argument("--max-tokens", type=int, default=256,
                       help="最大生成token数")
    parser.add_argument("--n-ctx", type=int, default=4096,
                       help="上下文长度")

    # 输出配置
    parser.add_argument("--output",
                       default="qwen_gptq_human_eval.jsonl",
                       help="输出文件名")

    args = parser.parse_args()

    print("=" * 60)
    print("HumanEval GPTQ量化模型评测")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"样本数量/问题: {args.samples}")
    print(f"温度: {args.temperature}")
    print(f"上下文长度: {args.n_ctx}")
    print(f"输出文件: {args.output}")
    print("=" * 60)

    # 初始化评测器
    try:
        evaluator = GPTQHumanEvalEvaluator(
            model_path=args.model_path,
            n_ctx=args.n_ctx
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