import os
from typing import Optional


class Config:
    """配置管理类"""
    
    @staticmethod
    def get_openai_api_key() -> Optional[str]:
        """获取OpenAI API密钥"""
        return os.getenv("OPENAI_API_KEY")
    
    @staticmethod
    def get_anthropic_api_key() -> Optional[str]:
        """获取Anthropic API密钥"""
        return os.getenv("ANTHROPIC_API_KEY")
    
    @staticmethod
    def get_gemini_api_key() -> Optional[str]:
        """获取Google Gemini API密钥"""
        return os.getenv("GEMINI_API_KEY")
    
    @staticmethod
    def get_deepseek_api_key() -> Optional[str]:
        """获取DeepSeek API密钥"""
        return os.getenv("DEEPSEEK_API_KEY")
    
    @staticmethod
    def get_ollama_base_url() -> str:
        """获取Ollama基础URL"""
        return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    @staticmethod
    def get_default_model() -> str:
        """获取默认模型"""
        return os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    
    @staticmethod
    def get_rate_limit_delay() -> float:
        """获取API调用间隔时间（秒）"""
        return float(os.getenv("RATE_LIMIT_DELAY", "1.0"))
    
    @staticmethod
    def get_max_content_length() -> int:
        """获取最大内容长度"""
        return int(os.getenv("MAX_CONTENT_LENGTH", "8000"))
    
    @staticmethod
    def get_batch_size() -> int:
        """获取批处理大小"""
        return int(os.getenv("BATCH_SIZE", "10"))