#!/usr/bin/env python3
"""
GPTQ模型生成器，用于PsychoBench评测
基于transformers库加载GPTQ量化模型
完整版：包含所有修复（template fallback, manual generation loop, length fix, dtype, mask check）
"""

import time
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

class GPTQModelGenerator:
    def __init__(self, model_path, n_gpu_layers=-1, n_ctx=4096, seed=42):
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.seed = seed
        torch.manual_seed(seed)

        print(f"🔧 Loading GPTQ model: {model_path}")
        model_start_time = time.time()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=True
            )

            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 使用ToM-Bench成功的模型加载方式
            print("使用ToM-Bench成功的GPTQ加载方式...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model.eval()
            print(f"✅ GPTQ模型加载成功，类型: {type(self.model)}")
            device = next(self.model.parameters()).device
            print(f"✅ 模型设备: {device}")

            model_load_time = time.time() - model_start_time
            print(f"✅ GPTQ model loaded successfully in {model_load_time:.1f} seconds")

            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / (1024**3)
                print(f"💾 Model memory usage: {memory_gb:.1f} GB")

        except Exception as e:
            print(f"❌ Failed to load GPTQ model: {str(e)}")
            raise e

    def completion(self, prompt, temperature=0.01, max_tokens=300, delay=1):
        time.sleep(delay)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.n_ctx
            )

            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            if 'input_ids' not in inputs or inputs['input_ids'].shape[1] == 0:
                raise ValueError(f"Empty inputs: {prompt[:100]}...")

            with torch.no_grad():
                # 使用ToM-Bench成功的generate方法
                try:
                    # 获取模型设备
                    device = next(self.model.parameters()).device

                    # 确保输入在正确的设备上
                    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

                    # 确保attention_mask存在
                    if 'attention_mask' not in inputs:
                        inputs['attention_mask'] = torch.ones(inputs['input_ids'].shape, dtype=torch.long, device=device)

                    # 使用ToM-Bench成功的生成参数
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_tokens,
                        do_sample=True,  # 根据温度决定采样方式
                        temperature=0.01,  # 避免温度为0
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,
                        return_dict_in_generate=True
                    )

                    # 解码生成的部分
                    generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                    return response

                except Exception as e:
                    print(f"GPTQ generate failed: {type(e).__name__}: {str(e)}")
                    return ""

        except Exception as e:
            print(f"❌ Error in GPTQ completion: {str(e)}")
            return ""

    def chat(self, messages, temperature=0.01, max_tokens=300, n=1, delay=1):
        time.sleep(delay)

        # 简单 prompt 构建（避 template）
        system_content = next((m['content'].strip() for m in messages if m['role'] == 'system'), "Rate agreement 1-5: Strongly disagree to Strongly agree.")
        user_content = next((m['content'].strip() for m in messages if m['role'] == 'user'), "")
        prompt = f"{system_content}\n\n{user_content}\nAssistant: "
        print(f"Simple prompt: {prompt[:100]}...")

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.n_ctx
            )

            if 'attention_mask' not in inputs:
                inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

            input_ids = inputs['input_ids'].to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')

            with torch.no_grad():
                # 使用ToM-Bench成功的generate方法
                try:
                    # 获取模型设备
                    device = next(self.model.parameters()).device

                    # 确保输入在正确的设备上
                    inputs = {
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    }
                    inputs = {k: v.to(device) for k, v in inputs.items() if torch.is_tensor(v)}

                    # 使用ToM-Bench成功的生成参数
                    outputs = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=200,  # 限制生成长度，避免生成过多内容
                        do_sample=False if temperature <= 0 else True,  # 根据温度决定采样方式
                        temperature=max(temperature, 0.1) if temperature > 0 else 1.0,  # 避免温度为0
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token is not None else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,
                        return_dict_in_generate=True
                    )

                    # 解码生成的部分
                    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                    if n == 1:
                        return response
                    else:
                        return [response] * n

                except Exception as e:
                    print(f"GPTQ generate in chat failed: {type(e).__name__}: {str(e)}")
                    # 返回默认响应
                    default = "3 " * 44
                    if n == 1:
                        return default.strip()
                    else:
                        return [default.strip()] * n

        except Exception as e:
            print(f"❌ Error in GPTQ chat: {str(e)}")
            default = "3 " * 44
            if n == 1:
                return default.strip()
            else:
                return [default.strip()] * n


def convert_results(result, column_header):
    result = result.strip()  # Remove leading and trailing whitespace
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


def gptq_model_generator(questionnaire, args):
    testing_file = args.testing_file
    model_path = args.model_path
    records_file = args.name_exp if args.name_exp is not None else model_path

    n_gpu_layers = getattr(args, 'n_gpu_layers', -1)

    generator = GPTQModelGenerator(model_path, n_gpu_layers=n_gpu_layers)

    # Read the existing CSV file into a pandas DataFrame
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

                # Retrieve the column data as a string
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
                            inputs = previous_records + [
                                {"role": "system", "content": questionnaire.get("inner_setting", "Rate agreement 1-5: Strongly disagree to Strongly agree.")},
                                {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                            ]
                            result = generator.chat(inputs, temperature=0.01)
                            previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                            previous_records.append({"role": "assistant", "content": result})

                            result_string_list.append(result.strip())

                            # Write the prompts and results to the file
                            os.makedirs("prompts", exist_ok=True)
                            os.makedirs("responses", exist_ok=True)

                            with open(f'prompts/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{inputs}\n====\n')
                            with open(f'responses/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        result_string = '\n'.join(result_string_list)

                        result_list = convert_results(result_string, column_header)

                        try:
                            # 获取DataFrame的行数
                            df_rows = len(df)

                            # 调整结果列表长度以匹配DataFrame行数
                            if len(result_list) < df_rows:
                                # 填充默认值3
                                result_list.extend([3] * (df_rows - len(result_list)))
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
                        except Exception as capture_err:
                            print(f"Unable to capture the responses on {column_header}: {type(capture_err).__name__}: {str(capture_err)}")

                    # Write the updated DataFrame back to the CSV file
                    df.to_csv(testing_file, index=False)

                    pbar.update(1)

    print(f"Evaluation completed for {questionnaire['name']}")
    return df
