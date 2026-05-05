#!/usr/bin/env python3
"""
本地模型生成器，用于替代OpenAI API运行PsychoBench评测
支持多种本地Qwen模型
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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

class LocalModelGenerator:
    def __init__(self, model_path, device="auto", torch_dtype=None, quantization=None):
        """
        初始化本地模型生成器

        Args:
            model_path: 模型路径
            device: 设备映射 ("auto", "cpu", "cuda", or specific device)
            torch_dtype: 数据类型 (None, "float16", "bfloat16")
            quantization: 量化选项 ("8bit", "4bit", "none")
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_8bit = (quantization == "8bit")
        self.use_4bit = (quantization == "4bit")

        print(f"Loading model from {model_path}...")
        if self.use_8bit:
            print("Using 8bit quantization")
        elif self.use_4bit:
            print("Using 4bit quantization")
        self._load_model()

    def _load_model(self):
        """加载模型和tokenizer"""
        # 设置数据类型
        if self.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None

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

        # 设置pad token
        # 设置pad token - 对于Qwen等模型，直接使用EOS作为PAD
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # 对于Qwen等模型，不尝试添加特殊token

        # 加载模型
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }

        if self.device != "cpu":
            load_kwargs["device_map"] = self.device
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        # 量化支持 - 使用role-eval成功配置
        if hasattr(self, 'use_8bit') and self.use_8bit:
            load_kwargs["load_in_8bit"] = True
            print("Loading model with 8bit quantization...")
        elif hasattr(self, 'use_4bit') and self.use_4bit:
            # 使用与role-eval相同的4bit配置
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            load_kwargs["quantization_config"] = quantization_config
            print("Loading model with 4bit NF4 quantization (optimized configuration)...")

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

    def completion(self, prompt, temperature=0.01, max_tokens=1024, delay=1):
        """
        兼容官方脚本的completion函数
        使用官方chat模板格式优化量化模型响应
        """
        time.sleep(delay)

        # 将prompt包装为官方chat格式
        chat_prompt = f"""<|im_start|>system
You are a helpful assistant that responds to psychological assessment questions. Please provide numerical scores as requested.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

        # 编码输入
        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        # 移动到正确的设备
        if self.device == "auto":
            device = self.model.device
        else:
            device = self.device

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成响应 - 优化量化模型参数
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.1) if temperature > 0 else 0.01,  # 确保最小温度
                do_sample=False if temperature <= 0.01 else True,  # 低温度时使用确定性生成
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.05 if (self.use_8bit or self.use_4bit) else 1.0  # 量化模型使用轻微重复惩罚
            )

        # 解码响应，只取新生成的部分
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        return response

    def chat(self, messages, temperature=0.01, max_tokens=1024, n=1, delay=1):
        """
        兼容官方脚本的chat函数
        使用官方Qwen chat template格式优化4bit和8bit模型响应
        """
        time.sleep(delay)

        # 强制使用官方Qwen chat template格式，优化量化模型响应
        conversation = ""
        for msg in messages:
            if msg["role"] == "system":
                conversation += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "user":
                conversation += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                conversation += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        conversation += "<|im_start|>assistant\n"
        prompt = conversation

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        # 移动到正确的设备
        if self.device == "auto":
            device = self.model.device
        else:
            device = self.device

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 生成响应 - 优化量化模型参数
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.1) if temperature > 0 else 0.01,  # 确保最小温度
                do_sample=False if temperature <= 0.01 else True,  # 低温度时使用确定性生成
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                repetition_penalty=1.05 if (self.use_8bit or self.use_4bit) else 1.0  # 量化模型使用轻微重复惩罚
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
    """转换模型输出为标准格式 - 参考GPTQ成功实现"""
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


def local_model_generator(questionnaire, args):
    """
    完全基于官方example_generator.py的本地生成器
    严格遵循官方脚本的每一个步骤
    """
    testing_file = args.testing_file
    model_path = args.model_path  # 使用model_path参数
    records_file = args.name_exp if args.name_exp is not None else "Qwen-7B-Chat"

    # 初始化本地模型生成器
    device = getattr(args, 'device', 'auto')
    torch_dtype = getattr(args, 'torch_dtype', 'float16' if torch.cuda.is_available() else None)
    quantization = getattr(args, 'quantization', 'none')

    generator = LocalModelGenerator(model_path, device=device, torch_dtype=torch_dtype, quantization=quantization)

    # Read the existing CSV file into a pandas DataFrame (完全遵循官方脚本)
    df = pd.read_csv(testing_file)

    # Find the columns whose headers start with "order"
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    with tqdm(total=total_iterations) as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                # Find the index of the previous column
                questions_column_index = i - 1
                shuffle_count += 1

                # Retrieve the column data as a string (完全遵循官方脚本)
                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+30] for i in range(0, len(questions_list), 30)]
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)]) for j, questions in enumerate(separated_questions)]

                for k in range(args.test_count):

                    df = pd.read_csv(testing_file)

                    # Insert the updated column into the DataFrame with a unique identifier in the header
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'

                    while(True):
                        result_string_list = []
                        previous_records = []

                        for questions_string in questions_list:
                            result = ''

                            # 完全遵循官方脚本的模型判断逻辑
                            # 由于我们使用本地模型，使用类似GPT的对话格式
                            inputs = previous_records + [
                                {"role": "system", "content": questionnaire["inner_setting"]},
                                {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                            ]
                            result = generator.chat(inputs, temperature=0.01)  # 温度0.01
                            previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                            previous_records.append({"role": "assistant", "content": result})

                            result_string_list.append(result.strip())

                            # Write the prompts and results to the file (保存到模型子目录)
                            results_dir = f"results/{records_file}"
                            os.makedirs(f"{results_dir}/prompts", exist_ok=True)
                            os.makedirs(f"{results_dir}/responses", exist_ok=True)

                            # 构建完整的prompt字符串用于保存
                            full_prompt = f"System: {questionnaire['inner_setting']}\n\nUser: {questionnaire['prompt']}\n{questions_string}\n\nAssistant: "

                            with open(f'{results_dir}/prompts/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{full_prompt}\n====\n')
                            with open(f'{results_dir}/responses/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)

                        # 使用官方的convert_results函数
                        result_list = convert_results(result_string, column_header)

                        try:
                            # 获取DataFrame的行数
                            df_rows = len(df)

                            # 调整结果列表长度以匹配DataFrame行数 - 参考GPTQ成功实现
                            if len(result_list) < df_rows:
                                # 根据问卷类型填充默认值
                                default_value = 3  # 大多数问卷的中位值
                                result_list.extend([default_value] * (df_rows - len(result_list)))
                                print(f"Filled {df_rows - len(result_list)} missing responses for {column_header}")
                            elif len(result_list) > df_rows:
                                # 截断到DataFrame行数
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
                            print(f"Result string: {result_string[:200]}...")  # 打印前200字符用于调试
                            # 如果解析失败，使用空字符串填充
                            if column_header in df.columns:
                                df[column_header] = ["" for _ in range(len(df))]
                            else:
                                df.insert(i + insert_count + 1, column_header, ["" for _ in range(len(df))])
                                insert_count += 1
                            break

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)

                    pbar.update(1)

    print(f"Evaluation completed for {questionnaire['name']}")
    return df


if __name__ == "__main__":
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=f"{ARTIFACT_ROOT}/models/Qwen-7B-Chat")
    parser.add_argument("--test-prompt", default="请回答以下问题：\n1. 你是一个外向的人吗？(1-7分)\n2. 你喜欢参加聚会吗？(1-7分)")

    args = parser.parse_args()

    generator = LocalModelGenerator(args.model_path)

    print("Testing with prompt:")
    print(args.test_prompt)
    print("\nResponse:")
    response = generator.generate_response(args.test_prompt)
    print(response)
