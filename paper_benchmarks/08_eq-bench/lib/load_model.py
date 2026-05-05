from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os

def load_model(base_model_path, lora_path, quantization, trust_remote_code = False):
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)

	# This is for llama2 models, but doesn't seem to have
	# adverse effects on benchmarks for other models.
	# ! Update: This appears to be no longer necessary for llama2 models. It causes issues with some models like THUDM/glm-4-9b-chat.
	#tokenizer.pad_token = tokenizer.eos_token
	#tokenizer.padding_side = "right"

	# Quantization Config
	if quantization == '4bit':
		# load as 4 bit
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16,
			bnb_4bit_use_double_quant=False
		)
	elif quantization == '8bit':
		# load as 8 bit
		quant_config = BitsAndBytesConfig(
			load_in_8bit=True,
		)
	else:
		quant_config = None

	# Check if this is a GPTQ model (by checking model path or config)
	is_gptq_model = 'gptq' in base_model_path.lower() or 'GPTQ' in base_model_path
	config_file = os.path.join(base_model_path, 'config.json')
	if os.path.exists(config_file):
		with open(config_file, 'r') as f:
			import json
			config = json.load(f)
			if 'quantization_config' in config:
				qc = config['quantization_config']
				if qc.get('quant_method') == 'gptq':
					is_gptq_model = True

	# Model loading
	if is_gptq_model and quant_config is None:
		# Load GPTQ model using auto-gptq
		try:
			from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
			base_model = AutoGPTQForCausalLM.from_quantized(
				base_model_path,
				device_map="auto",
				trust_remote_code=trust_remote_code,
				use_safetensors=True
			)
			# GPTQ models need special handling for generation
			# Set model type to enable generation support
			if not hasattr(base_model, 'generation_config'):
				from transformers import GenerationConfig
				base_model.generation_config = GenerationConfig()
		except Exception as e:
			print(f"Warning: Failed to load with auto-gptq: {e}")
			print("Falling back to standard transformers loading...")
			base_model = AutoModelForCausalLM.from_pretrained(
				base_model_path,
				device_map="auto",
				trust_remote_code=trust_remote_code,
				torch_dtype=torch.float16
			)
	elif quant_config:
		base_model = AutoModelForCausalLM.from_pretrained(
			base_model_path,
			quantization_config=quant_config,
			device_map="auto",
			trust_remote_code=trust_remote_code
		)
	else:
		base_model = AutoModelForCausalLM.from_pretrained(
			base_model_path,
			device_map="auto",
			trust_remote_code=trust_remote_code,
			torch_dtype=torch.bfloat16
		)

	if lora_path:
		peft_model = PeftModel.from_pretrained(base_model, lora_path)
		return peft_model, tokenizer
	else:
		return base_model, tokenizer