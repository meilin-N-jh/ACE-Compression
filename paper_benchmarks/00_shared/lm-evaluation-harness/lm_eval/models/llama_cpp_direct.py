"""
直接使用 llama-cpp-python Python API 的 lm-eval 后端
无需 HTTP 服务器，直接调用底层库
"""

import logging
import numpy as np
import os
from typing import Optional, List, Tuple, Union
import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


eval_logger = logging.getLogger(__name__)


@register_model("llama_cpp_direct")
class LlamaCppDirectLM(LM):
    """
    直接使用 llama-cpp-python 的 LM 后端
    """
    
    def __init__(
        self,
        pretrained: str,
        n_gpu_layers: int = 40,
        n_ctx: int = 4096,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        logits_all: bool = True,
        dtype: str = "float16",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        
        self.pretrained = pretrained
        self.dtype_str = dtype
        self.device = device
        
        eval_logger.info(f"加载 GGUF 模型: {pretrained}")
        
        # 导入 llama-cpp-python
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python 未安装。请运行: pip install llama-cpp-python"
            )
        
        # 根据设备选择 GPU 层数
        if "cuda" in device:
            actual_n_gpu_layers = n_gpu_layers
        else:
            actual_n_gpu_layers = 0
        
        # 加载模型
        default_threads = os.cpu_count() or 4
        self.llm = Llama(
            model_path=pretrained,
            n_gpu_layers=actual_n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads or default_threads,
            logits_all=logits_all,
            verbose=False,
        )
        
        eval_logger.info(f"✓ 模型加载成功，上下文长度: {n_ctx}")
    
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """计算续文的对数似然"""
        results = []
        total = len(requests)
        
        for idx, request in enumerate(requests, 1):
            # request.args 是 (context, continuation) 元组
            context, continuation = request.args
            
            # 完整文本
            full_text = context + continuation
            context_tokens = self.llm.tokenize(context.encode(), add_bos=False)
            context_len = len(context_tokens)
            
            # 仅做打分：必须禁止生成，并回显 prompt 才能拿到 prompt token 的 logprobs
            output = self.llm(
                full_text,
                max_tokens=0,
                echo=True,
                temperature=0.0,
                logprobs=1,  # top-1 足够判断 greedy，开更大只会更慢
            )
            
            # 提取 token logprobs
            # logprobs 格式: {"text_offset": [...], "tokens": [...], "token_logprobs": [...], "top_logprobs": [...]}
            logprobs_dict = output.get("logprobs", {})
            token_logprobs = logprobs_dict.get("token_logprobs", [])
            
            if not token_logprobs or len(token_logprobs) <= context_len:
                # 降级处理：如果没有足够的 logprobs，返回 0
                eval_logger.debug(
                    f"警告: 无法获取完整 logprobs (上下文长度: {context_len}, "
                    f"返回数: {len(token_logprobs)})"
                )
                results.append((0.0, False))
                continue
            
            # 累加续文部分的 logprobs (跳过上下文部分)
            continuation_logprobs = sum(token_logprobs[context_len:])
            
            # 判断是否为贪心预测
            is_greedy = self._is_greedy_continuation(
                logprobs_dict.get("top_logprobs", []),
                logprobs_dict.get("tokens", []),
                context_len
            )
            
            results.append((continuation_logprobs, is_greedy))

            if idx % 10 == 0 or idx == total:
                eval_logger.info(f"loglikelihood 进度: {idx}/{total} ({100*idx//total}%)")
        
        return results
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        """计算完整文本的滚动对数似然（用于困惑度）"""
        results = []
        
        for request in requests:
            text = request.args[0]
            
            # 分块处理（如果文本超过上下文长度）
            tokens = self.llm.tokenize(text.encode())
            max_ctx = self.llm.n_ctx()
            
            total_logprob = 0.0
            
            if len(tokens) <= max_ctx:
                # 短文本：直接处理
                output = self.llm(
                    text,
                    max_tokens=0,
                    echo=True,
                    temperature=0.0,
                    logprobs=1,
                )
                token_logprobs = output.get("logprobs", {}).get("token_logprobs", [])
                if token_logprobs:
                    # 跳过 BOS token 的 logprob
                    total_logprob = sum(token_logprobs[1:]) if len(token_logprobs) > 1 else 0.0
            else:
                # 长文本：分块处理，保证每个 token 被预测一次
                stride = max_ctx // 2
                pos = 0
                
                while pos < len(tokens):
                    # 确定当前块的范围
                    start = max(0, pos - stride)
                    end = min(len(tokens), start + max_ctx)
                    
                    # 提取该块对应的文本
                    chunk_tokens = tokens[start:end]
                    chunk_text = self.llm.detokenize(chunk_tokens).decode()
                    
                    # 获取该块的 logprobs
                    output = self.llm(
                        chunk_text,
                        max_tokens=0,
                        echo=True,
                        temperature=0.0,
                        logprobs=1,
                    )
                    chunk_logprobs = output.get("logprobs", {}).get("token_logprobs", [])
                    
                    if chunk_logprobs:
                        # 只记录该块对应部分的 logprobs（跳过前面的重叠部分）
                        overlap = pos - start
                        total_logprob += sum(chunk_logprobs[max(1, overlap):])
                    
                    pos = end
            
            results.append((total_logprob,))
        
        return results
    
    def generate_until(self, requests) -> List[str]:
        """生成续文直到遇到停止序列"""
        results = []
        total = len(requests)
        
        for idx, request in enumerate(requests, 1):
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            
            # 提取生成参数
            max_tokens = gen_kwargs.get("max_tokens", 128)
            stop = gen_kwargs.get("until", [])
            top_k = gen_kwargs.get("top_k", 40)
            top_p = gen_kwargs.get("top_p", 0.9)
            temperature = gen_kwargs.get("temperature", 0.7)
            
            # 使用 llama-cpp 生成
            output = self.llm(
                context,
                max_tokens=max_tokens,
                stop=stop if isinstance(stop, list) else [stop],
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            
            # 提取生成的文本（去除输入部分）
            generated_text = output["choices"][0]["text"]
            results.append(generated_text)
            
            # 每 10 个样本输出一次进度
            if idx % 10 == 0:
                eval_logger.info(f"生成进度: {idx}/{total} ({100*idx//total}%)")
        
        return results
    
    def _is_greedy_continuation(self, top_logprobs: List, tokens: List, context_len: int) -> bool:
        """判断续文是否为贪心预测"""
        if not top_logprobs or not tokens:
            return False
        
        # 检查续文部分的每个 token 是否为该位置的最高概率 token
        for i in range(context_len, len(tokens)):
            if i >= len(top_logprobs):
                break
            
            top_tokens = top_logprobs[i]
            if not top_tokens:
                continue
            
            # 获取最高概率的 token
            max_token = max(top_tokens.keys(), key=lambda x: top_tokens[x])
            if max_token != tokens[i]:
                return False
        
        return True
    
    @property
    def tokenizer_name(self) -> str:
        return "llama"
