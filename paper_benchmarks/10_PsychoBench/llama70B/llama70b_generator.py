#!/usr/bin/env python3
"""
Llama-70B 模型生成器，用于 PsychoBench 评测
支持 FP16, GPTQ-INT4, GPTQ-INT8, 4bit bitsandbytes 等多种配置
使用方式: python run_psychobench_llama70b.py --model-path <path> --questionnaire BFI --shuffle-count 1 --test-count 1
"""
from pathlib import Path

def _find_artifact_root() -> Path:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "configs/models.yaml").exists():
            return candidate
    raise FileNotFoundError("Could not locate artifact root (configs/models.yaml).")

ARTIFACT_ROOT = str(_find_artifact_root())


import torch
import os
import pandas as pd
import time
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")

# 复用模型，避免每个问卷重复加载导致显存碎片化
_GENERATOR_CACHE = {"key": None, "generator": None}


class Llama70BModelGenerator:
    def __init__(self, model_path, device="auto", torch_dtype="float16", quantization=None):
        """
        初始化 Llama-70B 模型生成器

        Args:
            model_path: 模型路径
            device: 设备映射 ("auto", "cuda", or specific device like "cuda:0")
            torch_dtype: 数据类型 ("float16", "bfloat16", "auto")
            quantization: 量化选项 ("8bit", "4bit", "gptq_int4", "gptq_int8", "none")
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_8bit = (quantization == "8bit")
        self.use_4bit = (quantization == "4bit")
        self.use_gptq = "gptq" in quantization if quantization else False
        print(f"Loading Llama-70B model from {model_path}...")
        if self.use_8bit:
            print("Using 8bit quantization")
        elif self.use_4bit:
            print("Using 4bit bitsandbytes quantization")
        elif self.use_gptq:
            print(f"Using GPTQ quantization: {quantization}")
        self._load_model()

    def _load_model(self):
        """加载模型和tokenizer"""
        # 设置数据类型
        if self.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif self.torch_dtype == "auto":
            torch_dtype = torch.float16  # GPTQ 模型使用 float16
        else:
            torch_dtype = torch.float16

        # 加载tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False
            )
        except Exception as e:
            print(f"Failed to load tokenizer locally: {e}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    local_files_only=False,
                    trust_remote_code=True,
                    use_fast=False
                )
            except Exception as e2:
                print(f"Failed to load tokenizer with remote fallback: {e2}")
                raise e2

        # 设置pad token - Llama模型
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print("Set pad_token to eos_token")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print("Added [PAD] special token")

        # 加载模型
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }

        if self.device != "cpu":
            load_kwargs["device_map"] = self.device

        # 量化配置
        if self.use_8bit:
            load_kwargs["load_in_8bit"] = True
            print("Loading model with 8bit quantization...")

        elif self.use_4bit:
            # 4bit bitsandbytes 配置
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            load_kwargs["quantization_config"] = quantization_config
            print("Loading model with 4bit NF4 quantization...")

        # GPTQ 模型自动检测量化
        if self.use_gptq:
            load_kwargs["torch_dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )
        except Exception as e:
            print(f"Failed to load model locally: {e}")
            # 尝试不使用local_files_only
            try:
                load_kwargs["local_files_only"] = False
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_kwargs
                )
            except Exception as e2:
                print(f"Failed to load model with remote fallback: {e2}")
                raise e2

        print(f"Model loaded successfully!")
        if hasattr(self.model, 'device'):
            print(f"Model device: {self.model.device}")
        if hasattr(self.model, 'dtype'):
            print(f"Model dtype: {self.model.dtype}")

    def chat(self, messages, temperature=0.01, max_tokens=1024, n=1, delay=1):
        """
        兼容官方脚本的 chat 函数
        使用 Llama-3.1 chat template 格式
        """
        time.sleep(delay)

        # 使用 tokenizer 的 apply_chat_template 方法
        # 这会自动应用正确的 Llama chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        # 移动到正确的设备
        if hasattr(self.model, 'device'):
            device = self.model.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成响应 - 优化量化模型参数
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.1) if temperature > 0 else 0.01,
                do_sample=False if temperature <= 0.01 else True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.05 if (self.use_8bit or self.use_4bit or self.use_gptq) else 1.0
            )

        # 解码响应，只取新生成的部分
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        if n == 1:
            return response
        else:
            return [response] * n


def convert_results(result, column_header):
    """转换模型输出为标准格式"""
    result = result.strip()
    try:
        import re
        result_list = []
        for line in result.splitlines():
            match = re.search(r'(-?\d+)\s*%?\s*$', line.strip())
            if match:
                result_list.append(int(match.group(1)))
    except:
        result_list = ["" for element in result.split('\n')]
        print(f"Unable to capture the responses on {column_header}.")

    return result_list


def llama70b_generator(questionnaire, args):
    """
    Llama-70B 模型的 PsychoBench 评测生成器
    兼容官方 example_generator 接口
    """
    import os
    testing_file = args.testing_file
    model_path = args.model_path
    records_file = args.name_exp if args.name_exp is not None else "Llama-70B"

    # 设置结果保存目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_base_dir = os.path.join(script_dir, "results")
    prompts_dir = os.path.join(results_base_dir, "prompts")
    responses_dir = os.path.join(results_base_dir, "responses")

    # 获取模型配置参数
    device = getattr(args, 'device', 'auto')
    torch_dtype = getattr(args, 'torch_dtype', 'float16')
    quantization = getattr(args, 'quantization', 'none')

    cache_key = (model_path, device, torch_dtype, quantization)
    if _GENERATOR_CACHE["key"] != cache_key or _GENERATOR_CACHE["generator"] is None:
        # 如果配置变化，先释放旧模型
        if _GENERATOR_CACHE["generator"] is not None:
            try:
                del _GENERATOR_CACHE["generator"].model
            except Exception:
                pass
            _GENERATOR_CACHE["generator"] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 初始化模型（复用同一进程的模型实例）
        _GENERATOR_CACHE["generator"] = Llama70BModelGenerator(
            model_path,
            device=device,
            torch_dtype=torch_dtype,
            quantization=quantization
        )
        _GENERATOR_CACHE["key"] = cache_key

    generator = _GENERATOR_CACHE["generator"]

    # 读取测试文件
    df = pd.read_csv(testing_file)

    # 找到所有 order 开头的列
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    print(f"\nStarting evaluation for {questionnaire['name']}")
    print(f"Total iterations: {total_iterations}")
    print(f"Shuffle count: {len(order_columns)}, Test count: {args.test_count}\n")

    with tqdm(total=total_iterations, desc=f"{questionnaire['name']}") as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                questions_column_index = i - 1
                shuffle_count += 1

                # 获取问题列表
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+30] for i in range(0, len(questions_list), 30)]
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)]) for j, questions in enumerate(separated_questions)]

                for k in range(args.test_count):
                    df = pd.read_csv(testing_file)
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'

                    while True:
                        result_string_list = []
                        previous_records = []

                        for questions_string in questions_list:
                            result = ''

                            # 构建消息 - 使用与 example_generator 相同的格式
                            inputs = previous_records + [
                                {"role": "system", "content": questionnaire["inner_setting"]},
                                {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                            ]
                            result = generator.chat(inputs, temperature=0.01)
                            previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                            previous_records.append({"role": "assistant", "content": result})

                            result_string_list.append(result.strip())

                            # 保存 prompts 和 responses 到 llama70B/results 目录
                            os.makedirs(prompts_dir, exist_ok=True)
                            os.makedirs(responses_dir, exist_ok=True)

                            with open(f'{prompts_dir}/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{inputs}\n====\n')
                            with open(f'{responses_dir}/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)
                        result_list = convert_results(result_string, column_header)

                        try:
                            df_rows = len(df)

                            # 调整结果列表长度
                            if len(result_list) < df_rows:
                                default_value = 3
                                result_list.extend([default_value] * (df_rows - len(result_list)))
                                print(f"Filled {df_rows - len(result_list)} missing responses for {column_header}")
                            elif len(result_list) > df_rows:
                                result_list = result_list[:df_rows]
                                print(f"Truncated {len(result_list) - df_rows} excess responses for {column_header}")

                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break
                        except Exception as e:
                            print(f"Unable to capture the responses on {column_header}. Error: {e}")
                            print(f"Result string: {result_string[:200]}...")
                            if column_header in df.columns:
                                df[column_header] = ["" for _ in range(len(df))]
                            else:
                                df.insert(i + insert_count + 1, column_header, ["" for _ in range(len(df))])
                                insert_count += 1
                            break

                    # 保存结果
                    df.to_csv(testing_file, index=False)
                    pbar.update(1)

    print(f"Evaluation completed for {questionnaire['name']}")
    return df


if __name__ == "__main__":
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=f"{ARTIFACT_ROOT}/models/Llama3-70b/Llama-3.1-70B-Instruct")
    parser.add_argument("--test-prompt", default="Please answer the following questions:\n1. Are you an extroverted person? (1-7 scale)\n2. Do you like parties? (1-7 scale)")

    args = parser.parse_args()

    generator = Llama70BModelGenerator(args.model_path)

    print("Testing with prompt:")
    print(args.test_prompt)
    print("\nResponse:")
    response = generator.chat([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": args.test_prompt}
    ])
    print(response)
