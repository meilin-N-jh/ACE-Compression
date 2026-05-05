#!/usr/bin/env python3
"""
Llama3.1-8B vLLM API模型生成器，用于 PsychoBench 评测
支持 FP16, BNB 4bit, AWQ, GPTQ INT4 等多种配置
使用vLLM OpenAI兼容API进行推理
参考 qwen2.5-14b 的实现方式
"""

import requests
import time
import os
import sys
from tqdm import tqdm
import warnings
import re
import pandas as pd
warnings.filterwarnings("ignore")

# 添加父目录到路径以导入官方模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# 尝试导入官方的convert_results函数作为备用
try:
    from example_generator import convert_results as official_convert_results
    _HAS_OFFICIAL_CONVERT = True
except ImportError:
    _HAS_OFFICIAL_CONVERT = False
    print("[WARNING] Could not import official convert_results, using regex fallback")


# 使用正则表达式提取数字（与qwen14b相同的逻辑）
def convert_results(result, column_header):
    """
    转换模型输出为分数列表
    参考qwen14b/llama70B的实现，使用正则表达式提取数字，能处理各种格式
    """
    result = result.strip()

    try:
        if _HAS_OFFICIAL_CONVERT:
            result_list = official_convert_results(result, column_header)
        else:
            raise RuntimeError("official convert_results unavailable")
    except:
        result_list = []
        for line in result.splitlines():
            match = re.search(r'(-?\d+)\s*%?\s*$', line.strip())
            if match:
                result_list.append(int(match.group(1)))

    return result_list


# 复用模型连接，避免重复初始化
_GENERATOR_CACHE = {"key": None, "generator": None}


class Llama31_8B_VLLMGenerator:
    def __init__(self, base_url, model_name, api_key="EMPTY", timeout=120):
        """
        初始化 Llama3.1-8B vLLM API 生成器
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

        print(f"Using Llama3.1-8B model via vLLM API: {model_name}")
        print(f"API endpoint: {self.base_url}")

        # 测试连接
        self._test_connection()

    def _test_connection(self):
        """测试vLLM服务器连接"""
        try:
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            print("✓ vLLM API server is ready")
        except Exception as e:
            print(f"✗ Failed to connect to vLLM server: {e}")
            raise

    def chat(self, messages, temperature=0, max_tokens=1024, n=1, delay=1):
        """
        兼容官方脚本的 chat 函数
        使用vLLM OpenAI兼容API
        """
        time.sleep(delay)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": max_tokens,
            "n": n
        }

        # 重试机制
        last_err = None
        for attempt in range(6):
            try:
                resp = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                resp.raise_for_status()
                data = resp.json()

                choices = data.get("choices", [])
                if not choices:
                    raise RuntimeError("No choices returned from server.")

                return choices[0]["message"]["content"].strip()

            except requests.exceptions.HTTPError as e:
                last_err = e
                # 打印详细错误信息用于调试
                if resp.status_code == 400:
                    try:
                        error_detail = resp.json()
                        print(f"Bad Request detail: {error_detail}")
                    except:
                        print(f"Bad Request text: {resp.text}")
                wait_time = min(60, 2 ** attempt)
                print(f"Request failed (attempt {attempt+1}/6): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                last_err = e
                wait_time = min(60, 2 ** attempt)
                print(f"Request failed (attempt {attempt+1}/6): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"Request failed after retries: {last_err}")


def get_llama31_8b_vllm_generator(base_url, model_name, api_key="EMPTY", timeout=120, force_reload=False):
    """
    获取或创建 Llama3.1-8B vLLM generator 实例（单例模式）
    """
    cache_key = f"{base_url}|{model_name}"

    if not force_reload and _GENERATOR_CACHE["key"] == cache_key:
        return _GENERATOR_CACHE["generator"]

    # 创建新的generator
    generator = Llama31_8B_VLLMGenerator(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        timeout=timeout
    )

    # 更新缓存
    _GENERATOR_CACHE["key"] = cache_key
    _GENERATOR_CACHE["generator"] = generator

    return generator


def llama31_8b_vllm_generator(questionnaire, args):
    """
    PsychoBench生成器函数，兼容官方接口
    参考 qwen2.5-14b 的实现

    Args:
        questionnaire: 问卷配置（包含name, prompt, inner_setting等）
        args: 命令行参数对象，需要包含:
            - base_url: vLLM API地址
            - model: 模型名称
            - testing_file: 测试文件路径
            - name_exp: 实验名称
            - shuffle_count: shuffle次数
            - test_count: 测试次数
    """
    # 设置模型特定的结果目录（与qwen14b相同，使用脚本所在目录）
    model_folder = args.name_exp if args.name_exp else args.model
    results_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    model_results_dir = os.path.join(results_base_dir, model_folder)

    # 重写args中的文件路径，让utils.py使用我们的路径
    args.testing_file = os.path.join(model_results_dir, f"{model_folder}-{questionnaire['name']}.csv")
    args.results_file = os.path.join(model_results_dir, f"{model_folder}-{questionnaire['name']}.md")
    args.figures_file = f"{model_folder}-{questionnaire['name']}.png"

    # 创建必要的目录
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_base_dir, "figures"), exist_ok=True)

    # 创建generator
    generator = get_llama31_8b_vllm_generator(
        base_url=args.base_url,
        model_name=args.model,
        api_key=getattr(args, 'api_key', 'EMPTY'),
        timeout=120
    )

    # 如果测试文件不存在，调用generate_testfile生成
    if not os.path.exists(args.testing_file):
        print(f"Testing file not found, generating: {args.testing_file}")
        from utils import generate_testfile
        generate_testfile(questionnaire, args)

    testing_file = args.testing_file
    records_file = model_folder

    df = pd.read_csv(testing_file)
    order_columns = [col for col in df.columns if col.startswith("order")]
    shuffle_count = 0
    insert_count = 0
    total_iterations = len(order_columns) * args.test_count

    print(f"\n开始评测问卷: {questionnaire['name']}")
    print(f"总迭代次数: {total_iterations} (shuffles: {len(order_columns)}, tests: {args.test_count})")

    with tqdm(total=total_iterations, desc=f"{questionnaire['name']}") as pbar:
        for i, header in enumerate(df.columns):
            if header in order_columns:
                questions_column_index = i - 1
                shuffle_count += 1

                questions_list = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [questions_list[i:i+30] for i in range(0, len(questions_list), 30)]
                questions_list = ['\n'.join([f"{i+1}.{q.split('.')[1]}" for i, q in enumerate(questions)])
                                  for questions in separated_questions]

                for k in range(args.test_count):
                    df = pd.read_csv(testing_file)
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'

                    while True:
                        result_string_list = []
                        previous_records = []

                        for questions_string in questions_list:
                            # 构造消息（增强system prompt以避免量化模型输出解释文本）
                            enhanced_system = questionnaire["inner_setting"] + " Do NOT provide any explanations, reasoning, or additional text. ONLY output the scores in the specified format."
                            messages = previous_records + [
                                {"role": "system", "content": enhanced_system},
                                {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                            ]

                            # 调用generator
                            result = generator.chat(messages, temperature=0, max_tokens=1024, n=1, delay=0)

                            previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                            previous_records.append({"role": "assistant", "content": result})

                            result_string_list.append(result.strip())

                            # 保存prompts和responses到模型目录
                            prompts_dir = os.path.join(model_results_dir, "prompts")
                            responses_dir = os.path.join(model_results_dir, "responses")
                            os.makedirs(prompts_dir, exist_ok=True)
                            os.makedirs(responses_dir, exist_ok=True)

                            with open(f'{prompts_dir}/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{messages}\n====\n')
                            with open(f'{responses_dir}/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        # 合并结果
                        result_string = '\n'.join(result_string_list)
                        result_list = convert_results(result_string, column_header)

                        # 调整结果列表长度
                        df_rows = len(df)
                        if len(result_list) < df_rows:
                            default_value = 3
                            result_list.extend([default_value] * (df_rows - len(result_list)))
                            print(f"Filled {df_rows - len(result_list)} missing responses for {column_header}")
                        elif len(result_list) > df_rows:
                            result_list = result_list[:df_rows]
                            print(f"Truncated {len(result_list) - df_rows} excess responses for {column_header}")

                        try:
                            if column_header in df.columns:
                                df[column_header] = result_list
                            else:
                                df.insert(i + insert_count + 1, column_header, result_list)
                                insert_count += 1
                            break
                        except Exception as e:
                            print(f"Unable to capture the responses on {column_header}. Error: {e}")

                    df.to_csv(testing_file, index=False)
                    pbar.update(1)

    print(f"问卷 {questionnaire['name']} 评测完成！\n")

    # 后处理：将图表文件从 results/figures 复制到模型目录
    try:
        figure_src = os.path.join(results_base_dir, "figures", args.figures_file)
        figure_dst = os.path.join(model_results_dir, args.figures_file)
        if os.path.exists(figure_src) and not os.path.exists(figure_dst):
            import shutil
            shutil.copy(figure_src, figure_dst)
            print(f"Copied figure to: {figure_dst}")
    except Exception as e:
        print(f"Note: Could not copy figure file: {e}")
