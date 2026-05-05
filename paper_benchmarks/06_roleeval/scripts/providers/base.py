from abc import ABC, abstractmethod
import torch

class BaseProvider(ABC):
    """Base class for model providers"""
    
    def __init__(self, model_config):
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        pass
    
    def unload_model(self):
        """Unload model and free memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        self.unload_model()
