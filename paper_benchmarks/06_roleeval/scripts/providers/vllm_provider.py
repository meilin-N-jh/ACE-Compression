import os
import requests
from .base import BaseProvider


class VLLMProvider(BaseProvider):
    """Provider for vLLM OpenAI-compatible server."""

    def load_model(self):
        self.base_url = self.config.get("base_url") or os.environ.get("VLLM_BASE_URL")
        self.model_name = self.config.get("model_name") or os.environ.get("VLLM_MODEL")

        if not self.base_url or not self.model_name:
            raise ValueError("vLLM base_url/model_name is required. Set in config or VLLM_BASE_URL/VLLM_MODEL.")

        if self.base_url.endswith("/"):
            self.base_url = self.base_url[:-1]

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        extra_body = self.config.get("extra_body", {})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.config.get("temperature", 0.0),
            "top_p": self.config.get("top_p", 1.0),
            "max_tokens": self.config.get("max_new_tokens", 5),
            "stream": False,
        }

        # Merge extra_body if provided
        if extra_body:
            payload.update(extra_body)

        stop = self.config.get("stop")
        if stop:
            payload["stop"] = stop

        guided_choice = self.config.get("guided_choice")
        if guided_choice:
            payload["guided_choice"] = guided_choice

        guided_regex = self.config.get("guided_regex")
        if guided_regex:
            payload["guided_regex"] = guided_regex

        resp = requests.post(url, json=payload, timeout=self.config.get("timeout", 120))
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return (message.get("content") or "").strip()