import torch
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from .base import BaseProvider

class TransformersProvider(BaseProvider):
    """Provider for Hugging Face Transformers models (FP16, 8-bit, 4-bit)"""
    
    def load_model(self):
        print(f"Loading model: {self.config['model_path']}")
        
        # Prepare quantization config if needed
        if self.config.get('load_in_4bit'):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=self.config.get('bnb_4bit_use_double_quant', True)
            )
        elif self.config.get('load_in_8bit'):
            quantization_config = None
        else:
            quantization_config = None
        
        # Determine torch_dtype
        if self.config.get('torch_dtype') == 'float16':
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_path'],
            torch_dtype=torch_dtype,
            device_map=self.config.get('device_map', 'auto'),
            load_in_8bit=self.config.get('load_in_8bit', False),
            quantization_config=quantization_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_path'],
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Check if model has chat method (like Qwen)
        self.has_chat = hasattr(self.model, 'chat')
        print(f"✓ Model loaded successfully (has_chat={self.has_chat})")
    
    def generate(self, prompt: str, gen_config: Dict[str, Any] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to model's device
        model_device = next(self.model.parameters()).device
        input_ids = inputs['input_ids'].to(model_device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                temperature=0.01,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
