import torch
from .base import BaseProvider

class AutoGPTQProvider(BaseProvider):
    """Provider for AutoGPTQ quantized models"""
    
    def load_model(self):
        print(f"Loading AutoGPTQ model: {self.config['model_path']}")
        
        try:
            from auto_gptq import AutoGPTQForCausalLM
            from transformers import AutoTokenizer
            
            self.model = AutoGPTQForCausalLM.from_quantized(
                self.config['model_path'],
                device="cuda:0",  # 使用cuda:0而不是device_map=auto
                use_triton=True,  # 启用Triton以提高速度
                trust_remote_code=True,  # 必须启用以支持Qwen的chat方法
                use_safetensors=True,
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['model_path'],
                trust_remote_code=True,
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✓ AutoGPTQ model loaded successfully")
        
        except ImportError as e:
            raise ImportError(f"auto-gptq not installed: {e}")
    
    def generate(self, prompt: str) -> str:
        try:
            # 优先尝试使用chat方法（部分模型支持）
            try:
                response, _ = self.model.chat(self.tokenizer, prompt, history=None)
                return response.strip()
            except Exception:
                # 如果chat失败，回退到generate方式
                pass
            
            # 构造更稳健的 chat prompt
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions with a single letter (A, B, C, or D). Provide only the answer letter."},
                    {"role": "user", "content": prompt},
                ]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                chat_prompt = prompt

            # Tokenize input
            inputs = self.tokenizer(chat_prompt, return_tensors='pt')
            
            # Move to model's device
            model_device = next(self.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Generate - 使用关键字参数方式调用
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,  # 展开所有输入作为关键字参数
                    max_new_tokens=self.config.get('max_new_tokens', 5),
                    temperature=self.config.get('temperature', 0.01),
                    top_p=self.config.get('top_p', 0.95),
                    do_sample=self.config.get('do_sample', False),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the generated part
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            return response.strip()
        
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            return ""
