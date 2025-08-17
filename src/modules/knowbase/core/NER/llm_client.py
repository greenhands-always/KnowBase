import json
import time
from typing import List, Dict, Any, Optional
import requests
from openai import OpenAI
import anthropic
import google.generativeai as genai
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """LLM提供商抽象基类"""
    
    @abstractmethod
    def extract_concepts(self, content: str) -> List[str]:
        """提取概念"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI提供商"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def extract_concepts(self, content: str) -> List[str]:
        prompt = """
        请从以下技术文章中提取重要的技术概念、术语、框架名称、库名称、算法名称等。
        要求：
        1. 提取的概念应该是有意义的技术术语
        2. 排除过于通用的词汇（如"技术","应用","数据"等）
        3. 保留专有名词和技术术语
        4. 返回JSON格式的概念列表
        
        文章内容：
        {content}
        
        请返回以下格式的JSON：
        {{
            "concepts": ["概念1", "概念2", "概念3", ...]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的技术概念提取助手，擅长从文章中提取精确的技术术语和概念。"},
                    {"role": "user", "content": prompt.format(content=content)}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            return self._parse_response(result)
            
        except Exception as e:
            print(f"OpenAI API调用失败: {e}")
            return []
    
    def _parse_response(self, response: str) -> List[str]:
        """解析API响应"""
        import re
        
        try:
            # 查找JSON格式
            json_match = re.search(r'\{[^{}]*"concepts"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                concepts = data.get("concepts", [])
                return [str(c).strip() for c in concepts if c.strip()]
            
            # 尝试直接提取列表
            list_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if list_match:
                items = re.findall(r'"([^"]+)"', list_match.group())
                return [item.strip() for item in items if item.strip()]
                
        except json.JSONDecodeError:
            pass
        
        return []


class AnthropicProvider(LLMProvider):
    """Anthropic提供商"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
    
    def extract_concepts(self, content: str) -> List[str]:
        prompt = f"""
        请从以下技术文章中提取重要的技术概念、术语、框架名称、库名称、算法名称等。
        要求：
        1. 提取的概念应该是有意义的技术术语
        2. 排除过于通用的词汇
        3. 保留专有名词和技术术语
        4. 仅返回概念列表，不要其他解释
        
        文章内容：
        {content}
        
        请返回一个JSON数组格式的概念列表：
        ["概念1", "概念2", "概念3", ...]
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = response.content[0].text
            return self._parse_response(result)
            
        except Exception as e:
            print(f"Anthropic API调用失败: {e}")
            return []
    
    def _parse_response(self, response: str) -> List[str]:
        """解析API响应"""
        import re
        
        try:
            # 直接解析JSON数组
            json_match = re.search(r'\[[^\[\]]*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [str(c).strip() for c in data if c.strip()]
                
        except json.JSONDecodeError:
            pass
        
        return []


class GeminiProvider(LLMProvider):
    """Google Gemini提供商"""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def extract_concepts(self, content: str) -> List[str]:
        prompt = f"""
        请从以下技术文章中提取重要的技术概念、术语、框架名称、库名称、算法名称等。
        要求：
        1. 提取的概念应该是有意义的技术术语
        2. 排除过于通用的词汇
        3. 保留专有名词和技术术语
        4. 仅返回概念列表
        
        文章内容：
        {content}
        
        请以JSON数组格式返回概念列表：["概念1", "概念2", "概念3", ...]
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text
            return self._parse_response(result)
            
        except Exception as e:
            print(f"Gemini API调用失败: {e}")
            return []
    
    def _parse_response(self, response: str) -> List[str]:
        """解析API响应"""
        import re
        
        try:
            # 直接解析JSON数组
            json_match = re.search(r'\[[^\[\]]*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [str(c).strip() for c in data if c.strip()]
                
        except json.JSONDecodeError:
            pass
        
        return []


class OllamaProvider(LLMProvider):
    """Ollama本地模型提供商"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1"):
        self.base_url = base_url
        self.model = model
    
    def extract_concepts(self, content: str) -> List[str]:
        prompt = f"""
        请从以下技术文章中提取重要的技术概念、术语、框架名称、库名称、算法名称等。
        要求：
        1. 提取的概念应该是有意义的技术术语
        2. 排除过于通用的词汇
        3. 保留专有名词和技术术语
        4. 仅返回概念列表
        
        文章内容：
        {content}
        
        请以JSON数组格式返回概念列表：["概念1", "概念2", "概念3", ...]
        """
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()["response"]
                return self._parse_response(result)
            else:
                print(f"Ollama API调用失败: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Ollama API调用失败: {e}")
            return []
    
    def _parse_response(self, response: str) -> List[str]:
        """解析API响应"""
        import re
        
        try:
            # 直接解析JSON数组
            json_match = re.search(r'\[[^\[\]]*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return [str(c).strip() for c in data if c.strip()]
                
        except json.JSONDecodeError:
            pass
        
        return []


class LLMClient:
    """统一的LLM客户端"""
    
    def __init__(self, provider: str = "openai", **kwargs):
        """
        初始化LLM客户端
        
        Args:
            provider: 提供商名称 (openai, anthropic, gemini, ollama)
            **kwargs: 提供商特定的参数
        """
        self.provider_name = provider.lower()
        
        if self.provider_name == "openai":
            self.provider = OpenAIProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gpt-4o-mini"),
                base_url=kwargs.get("base_url")
            )
        elif self.provider_name == "anthropic":
            self.provider = AnthropicProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "claude-3-haiku-20240307")
            )
        elif self.provider_name == "gemini":
            self.provider = GeminiProvider(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "gemini-pro")
            )
        elif self.provider_name == "ollama":
            self.provider = OllamaProvider(
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                model=kwargs.get("model", "llama3.1")
            )
        else:
            raise ValueError(f"不支持的提供商: {provider}")
    
    def extract_concepts(self, content: str) -> List[str]:
        """提取概念"""
        return self.provider.extract_concepts(content)