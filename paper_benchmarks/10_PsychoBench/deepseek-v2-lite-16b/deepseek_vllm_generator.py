#!/usr/bin/env python3
"""
DeepSeek-V2-Lite-16B vLLM API模型生成器，用于 PsychoBench 评测
支持 FP16, BNB 4bit, AWQ 等多种配置
使用vLLM OpenAI兼容API进行推理
"""

import requests
import time
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 添加父目录到路径以导入官方模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

REPO_ROOT = os.path.dirname(os.path.dirname(parent_dir))
DEEPSEEK_TOKENIZER_PATH = os.path.join(
    REPO_ROOT,
    "models",
    "DeepSeek-V2-Lite-Chat-16B",
    "DeepSeek-V2-Lite-Chat",
)
MAX_CONTEXT_TOKENS = 8192
REQUEST_MAX_TOKENS_CAP = 1024
CONTEXT_SAFETY_MARGIN = 128


def get_score_bounds(questionnaire):
    """PsychoBench 的 scale 字段对二分类问卷和 Likert 问卷定义不同。"""
    if questionnaire["scale"] == 1:
        return 0, 1
    return 1, questionnaire["scale"] - 1


def get_default_score(questionnaire):
    low, high = get_score_bounds(questionnaire)
    return (low + high) // 2


def extract_scores(result, questionnaire):
    """
    对齐官方的“只产出分数”目标，但兼容 DeepSeek 常见的逗号串/重复输出格式。
    优先抽取 `index: score` 中冒号后的分数，避免把题号误当成答案。
    """
    import re

    score_min, score_max = get_score_bounds(questionnaire)
    normalized = (
        result.replace("\r", "\n")
        .replace("：", ":")
        .replace("，", ",")
        .replace("；", ";")
        .strip()
    )

    def in_range(token):
        try:
            value = int(token)
        except ValueError:
            return None
        if score_min <= value <= score_max:
            return value
        return None

    extracted = []

    # 官方目标格式: `statement index: score`
    for match in re.finditer(r'(?<!\d)\d{1,3}\s*[:=]\s*(-?\d+)', normalized):
        value = in_range(match.group(1))
        if value is not None:
            extracted.append(value)
    if extracted:
        return extracted

    # 兼容 `1. 5` / `1) 5` 这类逐行输出。
    for match in re.finditer(r'(?<!\d)\d{1,3}\s*[\)\].-]\s*(-?\d+)', normalized):
        value = in_range(match.group(1))
        if value is not None:
            extracted.append(value)
    if extracted:
        return extracted

    # 退化场景：逐行只在末尾给一个分数。
    for line in normalized.splitlines():
        match = re.search(r'(-?\d+)\s*%?\s*$', line.strip())
        if not match:
            continue
        value = in_range(match.group(1))
        if value is not None:
            extracted.append(value)
    if extracted:
        return extracted

    # 最后兜底：纯分数列表 `1,2,3,...`
    for token in re.findall(r'(?<!\d)(-?\d+)(?!\d)', normalized):
        value = in_range(token)
        if value is not None:
            extracted.append(value)

    return extracted


# 复用模型连接，避免重复初始化
_GENERATOR_CACHE = {"key": None, "generator": None}
_TOKENIZER_CACHE = None


def get_deepseek_tokenizer():
    """按需加载 tokenizer，用于请求侧上下文预算控制。"""
    global _TOKENIZER_CACHE

    if _TOKENIZER_CACHE is None:
        from transformers import AutoTokenizer

        print(f"Loading DeepSeek tokenizer from: {DEEPSEEK_TOKENIZER_PATH}")
        _TOKENIZER_CACHE = AutoTokenizer.from_pretrained(
            DEEPSEEK_TOKENIZER_PATH,
            trust_remote_code=True,
        )

    return _TOKENIZER_CACHE


class DeepSeek_VLLMGenerator:
    def __init__(self, base_url, model_name, api_key="EMPTY", timeout=120):
        """
        初始化 DeepSeek vLLM API 生成器

        Args:
            base_url: vLLM服务器地址 (如 http://127.0.0.1:8300/v1)
            model_name: 模型名称 (如 deepseek-v2-lite-fp16)
            api_key: API密钥 (默认EMPTY)
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

        print(f"Using DeepSeek-V2-Lite model via vLLM API: {model_name}")
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
            print("vLLM API server is ready")
        except Exception as e:
            print(f"Failed to connect to vLLM server: {e}")
            raise

    def _resolve_safe_max_tokens(self, messages, requested_max_tokens):
        """确保 prompt tokens + max_tokens 不会超过 vLLM 的 max_model_len。"""
        tokenizer = get_deepseek_tokenizer()
        prompt_tokens = len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        available_tokens = MAX_CONTEXT_TOKENS - prompt_tokens - CONTEXT_SAFETY_MARGIN

        if available_tokens <= 0:
            raise RuntimeError(
                "Prompt exceeds context budget: "
                f"prompt_tokens={prompt_tokens}, "
                f"max_model_len={MAX_CONTEXT_TOKENS}, "
                f"safety_margin={CONTEXT_SAFETY_MARGIN}"
            )

        resolved_max_tokens = min(
            requested_max_tokens,
            REQUEST_MAX_TOKENS_CAP,
            available_tokens,
        )
        if resolved_max_tokens != requested_max_tokens:
            print(
                "Adjusted max_tokens "
                f"from {requested_max_tokens} to {resolved_max_tokens} "
                f"(prompt_tokens={prompt_tokens}, max_model_len={MAX_CONTEXT_TOKENS})"
            )

        return resolved_max_tokens

    def chat(self, messages, temperature=0, max_tokens=REQUEST_MAX_TOKENS_CAP, n=1, delay=1):
        """
        兼容官方脚本的 chat 函数
        使用vLLM OpenAI兼容API
        """
        time.sleep(delay)
        resolved_max_tokens = self._resolve_safe_max_tokens(messages, max_tokens)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": resolved_max_tokens,
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
                wait_time = min(60, 2 ** attempt)
                response_preview = getattr(e.response, "text", "")[:500]
                if response_preview:
                    print(f"HTTP error response: {response_preview}")
                print(f"Request failed (attempt {attempt+1}/6): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

            except Exception as e:
                last_err = e
                wait_time = min(60, 2 ** attempt)
                print(f"Request failed (attempt {attempt+1}/6): {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"Request failed after retries: {last_err}")


def get_deepseek_vllm_generator(base_url, model_name, api_key="EMPTY", timeout=120, force_reload=False):
    """
    获取或创建 DeepSeek vLLM generator 实例（单例模式）
    """
    cache_key = f"{base_url}|{model_name}"

    if not force_reload and _GENERATOR_CACHE["key"] == cache_key:
        return _GENERATOR_CACHE["generator"]

    # 创建新的generator
    generator = DeepSeek_VLLMGenerator(
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        timeout=timeout
    )

    # 更新缓存
    _GENERATOR_CACHE["key"] = cache_key
    _GENERATOR_CACHE["generator"] = generator

    return generator


def deepseek_vllm_generator(questionnaire, args):
    """
    PsychoBench生成器函数，兼容官方接口
    """
    import pandas as pd
    import re

    # 设置模型特定的结果目录
    model_folder = args.name_exp if args.name_exp else args.model
    results_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    model_results_dir = os.path.join(results_base_dir, model_folder)

    # 重写args中的文件路径，让utils.py使用我们的路径
    original_testing_file = args.testing_file
    args.testing_file = os.path.join(model_results_dir, f"{model_folder}-{questionnaire['name']}.csv")
    args.results_file = os.path.join(model_results_dir, f"{model_folder}-{questionnaire['name']}.md")
    args.figures_file = f"{model_folder}-{questionnaire['name']}.png"

    # 创建必要的目录
    os.makedirs(model_results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_base_dir, "figures"), exist_ok=True)

    # 创建generator
    generator = get_deepseek_vllm_generator(
        base_url=args.base_url,
        model_name=args.model,
        api_key=getattr(args, 'api_key', 'EMPTY'),
        timeout=120
    )

    # 如果测试文件不存在，从原始位置复制或生成
    if not os.path.exists(args.testing_file):
        print(f"Testing file not found, copying from: {original_testing_file}")
        if os.path.exists(original_testing_file):
            os.makedirs(os.path.dirname(args.testing_file), exist_ok=True)
            import shutil
            shutil.copy(original_testing_file, args.testing_file)
        else:
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

                questions_series = df.iloc[:, questions_column_index].astype(str)
                separated_questions = [
                    questions_series[i:i+30]
                    for i in range(0, len(questions_series), 30)
                ]
                question_blocks = [
                    '\n'.join([f"{i+1}.{q.split('.', 1)[1]}" for i, q in enumerate(questions)])
                    for questions in separated_questions
                ]
                question_counts = [len(questions) for questions in separated_questions]

                for k in range(args.test_count):
                    df = pd.read_csv(testing_file)
                    column_header = f'shuffle{shuffle_count - 1}-test{k}'

                    while True:
                        result_list = []
                        previous_records = []

                        for questions_string, expected_count in zip(question_blocks, question_counts):
                            # 构造消息
                            enhanced_system = (
                                questionnaire["inner_setting"]
                                + " Output exactly one answer per line in the format"
                                + ' "statement index: score".'
                                + " Do NOT output commas, explanations, summaries, or repeated prior answers."
                            )
                            messages = previous_records + [
                                {"role": "system", "content": enhanced_system},
                                {"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string}
                            ]

                            # 调用generator
                            result = generator.chat(
                                messages,
                                temperature=0,
                                max_tokens=REQUEST_MAX_TOKENS_CAP,
                                n=1,
                                delay=0,
                            )

                            previous_records.append({"role": "user", "content": questionnaire["prompt"] + '\n' + questions_string})
                            previous_records.append({"role": "assistant", "content": result})

                            chunk_scores = extract_scores(result, questionnaire)
                            if len(chunk_scores) > expected_count:
                                chunk_scores = chunk_scores[:expected_count]

                            if len(chunk_scores) < expected_count:
                                missing = expected_count - len(chunk_scores)
                                default_value = get_default_score(questionnaire)
                                chunk_scores.extend([default_value] * missing)
                                print(
                                    f"Filled {missing} missing responses in chunk for "
                                    f"{column_header} ({questionnaire['name']})"
                                )

                            result_list.extend(chunk_scores)

                            # 保存prompts和responses
                            prompts_dir = os.path.join(model_results_dir, "prompts")
                            responses_dir = os.path.join(model_results_dir, "responses")
                            os.makedirs(prompts_dir, exist_ok=True)
                            os.makedirs(responses_dir, exist_ok=True)

                            with open(f'{prompts_dir}/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{messages}\n====\n')
                            with open(f'{responses_dir}/{records_file}-{questionnaire["name"]}-shuffle{shuffle_count - 1}.txt', "a") as file:
                                file.write(f'{result}\n====\n')

                        # 调整结果列表长度
                        df_rows = len(df)
                        if len(result_list) < df_rows:
                            missing = df_rows - len(result_list)
                            default_value = get_default_score(questionnaire)
                            result_list.extend([default_value] * missing)
                            print(f"Filled {missing} missing responses for {column_header}")
                        elif len(result_list) > df_rows:
                            excess = len(result_list) - df_rows
                            result_list = result_list[:df_rows]
                            print(f"Truncated {excess} excess responses for {column_header}")

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

    # 后处理：将图表文件移动到模型目录
    try:
        figure_src = os.path.join(results_base_dir, "figures", args.figures_file)
        figure_dst = os.path.join(model_results_dir, args.figures_file)
        if os.path.exists(figure_src) and not os.path.exists(figure_dst):
            import shutil
            shutil.copy(figure_src, figure_dst)
            print(f"Copied figure to: {figure_dst}")
    except Exception as e:
        print(f"Note: Could not copy figure file: {e}")
