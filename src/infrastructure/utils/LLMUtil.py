from langchain_community.llms import Ollama
from langchain.llms.base import BaseLLM
from langchain_ollama import OllamaLLM
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI  # 更新导入
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Anthropic
import os
import requests
import json

from dotenv import load_dotenv
load_dotenv()


class LLMProvider:
    """LLM服务提供商抽象基类"""

    def get_llm(self, temperature=0.0, **kwargs):
        """获取LLM实例"""
        raise NotImplementedError

    def get_chat_model(self, temperature=0.0, **kwargs):
        """获取聊天模型实例"""
        raise NotImplementedError

    def list_models(self):
        """获取模型列表"""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI原生API提供商"""

    def get_llm(self, temperature=0.0, **kwargs):
        return ChatOpenAI(
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs
        )

    def get_chat_model(self, temperature=0.0, **kwargs):
        return ChatOpenAI(
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            **kwargs
        )

    def list_models(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        response.raise_for_status()
        return [model["id"] for model in response.json()["data"]]


class CompatibleOpenAIProvider(LLMProvider):
    """兼容OpenAI协议的第三方API提供商"""

    def __init__(self, api_base, api_key_env_var, api_version=None):
        self.api_base = api_base
        self.api_key_env_var = api_key_env_var

    def get_llm(self, model_name=None, temperature=0.0, **kwargs):
        # 移除可能冲突的 model 参数
        kwargs.pop('model', None)
        # 移除特殊参数，避免传递给ChatOpenAI构造函数
        kwargs.pop('enable_thinking', None)
        
        # 为阿里云等API设置特殊参数
        extra_body = {}
        if 'dashscope.aliyuncs.com' in self.api_base:
            extra_body['enable_thinking'] = False
        
        return ChatOpenAI(
            model=model_name or "gpt-3.5-turbo",
            temperature=temperature,
            base_url=self.api_base,
            api_key=os.getenv(self.api_key_env_var),
            extra_body=extra_body,
            **kwargs
        )

    def get_chat_model(self, model_name=None, temperature=0.0, **kwargs):
        # 移除可能冲突的 model 参数
        kwargs.pop('model', None)
        # 移除特殊参数，避免传递给ChatOpenAI构造函数
        kwargs.pop('enable_thinking', None)
        
        # 为阿里云等API设置特殊参数
        extra_body = {}
        if 'dashscope.aliyuncs.com' in self.api_base:
            extra_body['enable_thinking'] = False
        
        return ChatOpenAI(
            model=model_name or "gpt-3.5-turbo",
            temperature=temperature,
            base_url=self.api_base,
            api_key=os.getenv(self.api_key_env_var),
            extra_body=extra_body,
            **kwargs
        )

    def list_models(self):
        api_key = os.getenv(self.api_key_env_var)
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.get(f"{self.api_base}/models", headers=headers)
        response.raise_for_status()
        return [model["id"] for model in response.json()["data"]]


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Hub API提供商"""

    def get_llm(self, temperature=0.0, **kwargs):
        return HuggingFaceHub(
            repo_id=kwargs.get("model_name", "google/flan-t5-xl"),
            temperature=temperature,
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            **kwargs
        )

    def list_models(self):
        # Note: Hugging Face Hub doesn't have a simple "list all models" endpoint
        # like OpenAI. This is a placeholder for a more complex implementation
        # that might involve scraping or using a more specific API if available.
        return ["google/flan-t5-xl", "gpt2", "bert-base-uncased"]


class OllamaProvider(LLMProvider):
    """Ollama本地模型提供商"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def get_llm(self, temperature=0.0, **kwargs) -> BaseLLM:
        """获取Ollama LLM实例"""
        return OllamaLLM(  # 更新为 OllamaLLM
            model=self.model_name,
            base_url=self.base_url,
            temperature=temperature,
            **kwargs
        )

    def list_models(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return []

class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) API提供商"""

    def get_llm(self, temperature=0.0, **kwargs):
        return Anthropic(
            temperature=temperature,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            **kwargs
        )

    def list_models(self):
        # Placeholder: Anthropic's API for listing models might differ.
        # This is a common pattern, but requires checking their documentation.
        return ["claude-2", "claude-instant-1"]


# 工厂函数：根据名称获取提供商实例
def get_provider(provider_name, **kwargs):
    provider_name = provider_name.lower()
    if provider_name == "openai":
        return OpenAIProvider()
    elif provider_name == "azure":
        return CompatibleOpenAIProvider(
            api_base=kwargs.get("api_base", "https://your-azure-endpoint.openai.azure.com"),
            api_key_env_var="AZURE_OPENAI_API_KEY",
            api_version="2023-05-15"
        )
    elif provider_name == "localai":
        return CompatibleOpenAIProvider(
            api_base=kwargs.get("api_base", "http://localhost:8080/v1"),
            api_key_env_var="LOCALAI_API_KEY"
        )
    elif provider_name == "compatible":
        return CompatibleOpenAIProvider(
            api_base=kwargs.get("api_base", ""),
            api_key_env_var=kwargs.get("api_key", ""),
        )
    elif provider_name == "huggingface":
        return HuggingFaceProvider()
    elif provider_name == "claude":
        return AnthropicProvider()
    elif provider_name == "ollama":
        return OllamaProvider(
            model_name=kwargs.get("model_name", "mistral"),
            base_url=kwargs.get("base_url", "http://localhost:11434")
        )
    else:
        return None
