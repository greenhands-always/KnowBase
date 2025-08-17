"""
概念提取器模块
提供基于LLM的概念、实体和关键词提取功能
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI

# 尝试导入新版本的ChatOpenAI
try:
    from langchain_openai import ChatOpenAI as NewChatOpenAI
except ImportError:
    NewChatOpenAI = None

# 导入LLM工具类
from src.infrastructure.utils.LLMUtil import get_provider


class ConceptExtractionResult(BaseModel):
    """概念提取结果数据模型"""
    concepts: List[str] = Field(description="从文本中提取的关键概念列表", default_factory=list)
    entities: List[str] = Field(description="重要的实体名称", default_factory=list)
    keywords: List[str] = Field(description="核心关键词", default_factory=list)
    confidence: float = Field(description="提取结果的置信度", default=1.0)
    metadata: Dict[str, Any] = Field(description="额外的元数据", default_factory=dict)


class BaseConceptExtractor(ABC):
    """概念提取器抽象基类"""
    
    @abstractmethod
    def extract_from_text(self, text: str, **kwargs) -> ConceptExtractionResult:
        """从文本中提取概念"""
        pass
    
    @abstractmethod
    def extract_from_file(self, file_path: Union[str, Path], **kwargs) -> ConceptExtractionResult:
        """从文件中提取概念"""
        pass


class LLMConceptExtractor(BaseConceptExtractor):
    """基于LLM的概念提取器"""
    
    def __init__(self, llm, prompt_template: Optional[str] = None):
        """
        初始化LLM概念提取器
        
        Args:
            llm: LLM实例
            prompt_template: 自定义提示模板
        """
        self.llm = self._validate_llm(llm)
        self.prompt_template = prompt_template or self._get_default_prompt()
        self.parser = PydanticOutputParser(pydantic_object=ConceptExtractionResult)
        self._setup_chain()
    
    def _validate_llm(self, llm):
        """验证LLM实例"""
        valid_types = (OpenAI, ChatOpenAI)
        if NewChatOpenAI:
            valid_types = valid_types + (NewChatOpenAI,)
        
        # 检查是否有invoke方法（Langchain LLM的基本特征）
        if not hasattr(llm, 'invoke') and not isinstance(llm, valid_types):
            raise ValueError(f"llm 参数必须是支持Langchain接口的LLM实例，当前类型: {type(llm)}")
        
        return llm
    
    def _get_default_prompt(self) -> str:
        """获取默认的提示模板"""
        return """
        请从以下文档中提取关键概念、实体和关键词，并根据文章主题进行标签化。
        请注意处理标点符号和文字之间的关系，准确识别完整的概念。

        文档内容：
        {content}

        {format_instructions}

        请确保：
        1. 提取完整的概念短语，不要被标点符号截断
        2. 识别专业术语和技术概念
        3. 包含重要的实体名称（人名、地名、组织名等）
        4. 提取核心关键词
        5. 评估提取结果的置信度
        """
    
    def _setup_chain(self):
        """设置处理链"""
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["content"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser
    
    def extract_from_text(self, text: str, **kwargs) -> ConceptExtractionResult:
        """
        从文本中提取概念
        
        Args:
            text: 输入文本
            **kwargs: 额外参数
            
        Returns:
            ConceptExtractionResult: 提取结果
        """
        try:
            result = self.chain.invoke({"content": text})
            
            # 添加元数据
            result.metadata.update({
                "text_length": len(text),
                "extraction_method": "llm",
                "extraction_model": self.llm.model,
                "model_type": type(self.llm).__name__
            })
            result.metadata.update(kwargs)
            
            return result
            
        except Exception as e:
            print(f"文本概念提取时出错: {e}")
            return ConceptExtractionResult(
                confidence=0.0,
                metadata={"error": str(e), "extraction_method": "llm"}
            )
    
    def extract_from_file(self, file_path: Union[str, Path], encoding: str = 'utf-8', **kwargs) -> ConceptExtractionResult:
        """
        从文件中提取概念
        
        Args:
            file_path: 文件路径
            encoding: 文件编码
            **kwargs: 额外参数
            
        Returns:
            ConceptExtractionResult: 提取结果
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ConceptExtractionResult(
                confidence=0.0,
                metadata={"error": f"文件不存在: {file_path}", "file_path": str(file_path)}
            )
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            result = self.extract_from_text(content, **kwargs)
            
            # 添加文件相关元数据
            result.metadata.update({
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_name": file_path.name
            })
            
            return result
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return ConceptExtractionResult(
                confidence=0.0,
                metadata={
                    "error": str(e),
                    "file_path": str(file_path),
                    "extraction_method": "llm"
                }
            )


class ConceptExtractor:
    """概念提取器工厂类"""
    
    @staticmethod
    def create_llm_extractor(llm, prompt_template: Optional[str] = None) -> LLMConceptExtractor:
        """创建LLM概念提取器"""
        return LLMConceptExtractor(llm, prompt_template)
    
    @staticmethod
    def create_from_config(config: Dict[str, Any]) -> BaseConceptExtractor:
        """从配置创建概念提取器"""
        extractor_type = config.get("type", "llm")
        
        if extractor_type == "llm":
            llm = config["llm"]
            prompt_template = config.get("prompt_template")
            return LLMConceptExtractor(llm, prompt_template)
        else:
            raise ValueError(f"不支持的提取器类型: {extractor_type}")
    
    @staticmethod
    def create_llm_from_processing_config(processing_config) -> BaseConceptExtractor:
        """从处理配置创建LLM概念提取器"""
        try:
            # 根据配置创建LLM实例
            llm = ConceptExtractor._create_llm_instance(processing_config)
            return LLMConceptExtractor(llm)
        except Exception as e:
            raise ValueError(f"无法从配置创建LLM概念提取器: {e}")
    
    @staticmethod
    def _create_llm_instance(config):
        """根据配置创建LLM实例"""
        provider_name = config.llm_provider.lower()
        
        if provider_name == "ollama":
            provider = get_provider("ollama", 
                                  model_name=config.llm_model,
                                  base_url="http://localhost:11434")
            return provider.get_llm(temperature=config.llm_temperature,
                                  max_tokens=config.llm_max_tokens)
        
        elif provider_name == "openai":
            # 设置环境变量（如果配置中有API key）
            if config.openai_api_key:
                os.environ["OPENAI_API_KEY"] = config.openai_api_key
            
            provider = get_provider("openai")
            return provider.get_chat_model(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
        
        elif provider_name == "openai_compatible":
            # 设置环境变量（如果配置中有API key）
            if config.openai_api_key:
                os.environ["COMPATIBLE_API_KEY"] = config.openai_api_key
            
            provider = get_provider("compatible",
                                  api_base=config.openai_api_base,
                                  api_key="COMPATIBLE_API_KEY")
            return provider.get_chat_model(
                model_name=config.llm_model,  # 使用 model_name 而不是 model
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
        
        else:
            raise ValueError(f"不支持的LLM提供商: {provider_name}")