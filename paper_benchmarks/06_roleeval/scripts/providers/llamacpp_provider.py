from .base import BaseProvider

class LlamaCppProvider(BaseProvider):
    """Provider for llama.cpp GGUF models"""
    
    def load_model(self):
        print(f"Loading GGUF model: {self.config['model_path']}")
        
        try:
            from llama_cpp import Llama
            
            self.model = Llama(
                model_path=self.config['model_path'],
                n_gpu_layers=self.config.get('n_gpu_layers', 32),
                n_threads=self.config.get('n_threads', 8),
                verbose=False,
            )
            
            print("✓ GGUF model loaded successfully")
        
        except ImportError as e:
            raise ImportError(f"llama-cpp-python not installed: {e}")
    
    def generate(self, prompt: str) -> str:
        try:
            output = self.model(
                prompt,
                max_tokens=self.config.get('max_new_tokens', 8),
                temperature=self.config.get('temperature', 0.01),
                top_p=self.config.get('top_p', 0.95),
                echo=False,
            )
            
            if output and 'choices' in output and len(output['choices']) > 0:
                response = output['choices'][0]['text']
                return response.strip()
            else:
                print(f"Unexpected output format: {output}")
                return ""
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
