#!/usr/bin/env python3
"""
HumanEval GGUF模型评测脚本
基于role-eval成功的GGUF优化经验
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
import argparse
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用GPU1，与role-eval GGUF配置一致

try:
    from llama_cpp import Llama
except ImportError:
    print("错误: 请安装llama-cpp-python库")
    print("安装命令: pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu118")
    sys.exit(1)

class GGUFEvaluator:
    def __init__(self, model_path, n_gpu_layers=-1, n_ctx=4096):
        """初始化GGUF评测器"""
        self.model_path = model_path
        print(f"正在加载GGUF模型: {model_path}")

        try:
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,  # GPU层数，-1表示全部使用GPU
                n_ctx=n_ctx,               # 上下文长度
                verbose=False,
                seed=42
            )
            print(f"✓ GGUF模型加载成功 (GPU层数: {n_gpu_layers}, 上下文: {n_ctx})")
        except Exception as e:
            print(f"✗ GGUF模型加载失败: {e}")
            raise

    def create_optimized_prompt(self, problem_prompt):
        """
        创建优化的prompt，专门用于GGUF代码生成
        使用极简直接的prompt避免生成解释文本
        """
        # 极简prompt，只要求完成代码，避免任何解释
        optimized_prompt = f"""{problem_prompt}"""

        return optimized_prompt

    def generate_completion(self, prompt, temperature=0.1, max_tokens=256):
        """
        生成代码补全，优化参数避免语法错误
        """
        try:
            # 更保守的参数，专注于代码生成
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,  # 降低温度提高稳定性
                top_p=0.9,
                top_k=30,
                repeat_penalty=1.1,
                stop=["\ndef ", "\nclass ", "\n# ", "\nif __name__", "\n# ", "\n\"\"\"", "\n'''"],
                echo=False,
                stream=False
            )

            completion = response['choices'][0]['text'].strip()

            # 清理completion，移除解释和测试代码，标准化缩进
            lines = completion.split('\n')
            code_lines = []
            in_code_block = True

            for line in lines:
                # 跳过markdown和解释行
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if not in_code_block:
                    continue

                # 跳过注释和模板
                if line.strip().startswith('#') and any(word in line.lower() for word in ['your code here', 'example', 'usage', 'test', 'calculate']):
                    continue

                # 跳过包含测试代码的行
                if any(pattern in line.lower() for pattern in ['numbers = [', 'mad = mean_absolute_deviation', '```python', 'separate_paren_groups(', 'mean_absolute_deviation(']):
                    continue

                # 跳过明显的解释行
                if any(word in line.lower() for word in ['the function', 'this function', 'here is', 'output:', 'example:']):
                    continue

                # 保留有效的代码行
                if line.strip():  # 非空行
                    # 标准化缩进：统一为4个空格
                    stripped_line = line.lstrip()
                    if stripped_line:  # 确保strip后还有内容
                        code_lines.append('    ' + stripped_line)

            cleaned_completion = '\n'.join(code_lines)

            # 确保至少有一些内容
            if not cleaned_completion:
                cleaned_completion = "    pass"

            return cleaned_completion

        except Exception as e:
            print(f"生成补全时出错: {e}")
            return "pass"

    def evaluate_problems(self, num_samples_per_task=1, temperature=0.3):
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
                    # 创建优化的prompt
                    optimized_prompt = self.create_optimized_prompt(problem['prompt'])

                    # 生成补全
                    completion = self.generate_completion(
                        optimized_prompt,
                        temperature=temperature
                    )

                    # 保存样本
                    samples.append({
                        'task_id': task_id,
                        'completion': completion
                    })

                    pbar.update(1)

                    # 可选：打印示例进度
                    if len(samples) % 10 == 0:
                        print(f"\n已完成 {len(samples)} 个样本...")
                        print(f"最新样本 task_id: {task_id}")
                        print(f"生成补全: {completion[:100]}...")

        return samples

def main():
    parser = argparse.ArgumentParser(description="HumanEval GGUF模型评测")

    # 模型配置
    parser.add_argument("--model-path",
                       default=f"{ARTIFACT_ROOT}/models/Qwen-7B-Chat-GGUF/Qwen-7B-Chat.Q4_K_M.gguf",
                       help="GGUF模型路径")

    # 评测配置
    parser.add_argument("--samples", type=int, default=1,
                       help="每个问题的样本数量")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="生成温度")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--n-gpu-layers", type=int, default=35,
                       help="GPU层数")
    parser.add_argument("--n-ctx", type=int, default=4096,
                       help="上下文长度")

    # 输出配置
    parser.add_argument("--output",
                       default="qwen_gguf_human_eval_samples.jsonl",
                       help="输出文件名")

    args = parser.parse_args()

    print("=" * 60)
    print("HumanEval GGUF模型评测")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"样本数量/问题: {args.samples}")
    print(f"温度: {args.temperature}")
    print(f"GPU层数: {args.n_gpu_layers}")
    print(f"上下文长度: {args.n_ctx}")
    print(f"输出文件: {args.output}")
    print("=" * 60)

    # 初始化评测器
    try:
        evaluator = GGUFEvaluator(
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
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
    print(f"evaluate_functional_correctness {args.output}")

    return 0

if __name__ == "__main__":
    exit(main())