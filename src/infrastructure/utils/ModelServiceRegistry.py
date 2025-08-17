import os
import yaml
from typing import Dict, Any, Optional
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models import ChatOpenAI

from src.infrastructure.utils.LLMUtil import CompatibleOpenAIProvider


class ModelServiceRegistry:
    """模型服务注册表，用于管理和发现不同厂商的模型服务"""

    def __init__(self):
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        self.provider_factories = {
            "openai": self._create_openai_provider,
            "aliyun": self._create_aliyun_provider,
            "volcano": self._create_volcano_provider,
            # 其他提供商工厂函数...
        }

    def register_from_config(self, config_path: str) -> None:
        """从配置文件注册模型服务"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        for model_name, model_config in config.get("models", {}).items():
            self.register_model(model_name, model_config)

    def register_model(self, model_name: str, config: Dict[str, Any]) -> None:
        """注册单个模型服务"""
        provider_type = config.get("provider_type")
        if not provider_type:
            raise ValueError(f"模型 {model_name} 的配置缺少 provider_type")

        if provider_type not in self.provider_factories:
            raise ValueError(f"不支持的提供商类型: {provider_type}")

        # 创建提供商实例
        provider_factory = self.provider_factories[provider_type]
        provider = provider_factory(config)

        # 注册模型到注册表
        self.model_registry[model_name] = {
            "provider": provider,
            "config": config,
            "model_name": config.get("model_name", model_name)
        }

    def get_model(self, model_name: str) -> BaseLLM:
        """获取模型服务实例"""
        if model_name not in self.model_registry:
            raise ValueError(f"未注册的模型: {model_name}")

        model_info = self.model_registry[model_name]
        provider = model_info["provider"]
        model_config = model_info["config"]

        # 获取LLM实例，传递模型特定配置
        return provider.get_llm(
            temperature=model_config.get("temperature", 0.0),
            model_name=model_info["model_name"],
            **model_config.get("extra_args", {})
        )

    def _create_openai_provider(self, config: Dict[str, Any]):
        """创建OpenAI提供商实例"""
        return OpenAIProvider()

    def _create_aliyun_provider(self, config: Dict[str, Any]):
        """创建阿里云提供商实例"""
        return CompatibleOpenAIProvider(
            api_base=config["api_base"],
            api_key_env_var=config["api_key_env_var"],
            api_version=config.get("api_version")
        )

    def _create_volcano_provider(self, config: Dict[str, Any]):
        """创建火山引擎提供商实例"""
        return CompatibleOpenAIProvider(
            api_base=config["api_base"],
            api_key_env_var=config["api_key_env_var"],
            api_version=config.get("api_version")
        )