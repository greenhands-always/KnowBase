"""
向量化服务
支持多种向量化模型和方法
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """向量化配置"""
    model_name: str = "all-MiniLM-L6-v2"  # 默认使用sentence-transformers模型
    model_type: str = "sentence_transformers"  # sentence_transformers, transformers, openai
    vector_size: int = 384  # all-MiniLM-L6-v2的向量维度
    batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"  # cpu, cuda
    
    # OpenAI配置
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # 额外配置
    extra_config: Optional[Dict[str, Any]] = None


class EmbeddingService(ABC):
    """向量化服务抽象基类"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
    
    @abstractmethod
    async def load_model(self) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表"""
        pass
    
    @abstractmethod
    async def encode_text(self, text: str) -> List[float]:
        """将单个文本转换为向量"""
        pass
    
    async def get_vector_size(self) -> int:
        """获取向量维度"""
        return self.config.vector_size


class SentenceTransformersService(EmbeddingService):
    """基于sentence-transformers的向量化服务"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers未安装，请运行: pip install sentence-transformers")
    
    async def load_model(self) -> bool:
        """加载sentence-transformers模型"""
        try:
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device
            )
            
            # 更新向量维度
            self.config.vector_size = self._model.get_sentence_embedding_dimension()
            
            logger.info(f"成功加载sentence-transformers模型: {self.config.model_name}")
            logger.info(f"向量维度: {self.config.vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"加载sentence-transformers模型失败: {e}")
            return False
    
    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        if not self._model:
            raise RuntimeError("模型未加载")
        
        try:
            # 在线程池中运行编码，避免阻塞事件循环
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                self._model.encode,
                texts,
                self.config.batch_size,
                True,  # show_progress_bar
                True   # convert_to_numpy
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"批量编码文本失败: {e}")
            return []
    
    async def encode_text(self, text: str) -> List[float]:
        """编码单个文本"""
        results = await self.encode_texts([text])
        return results[0] if results else []


class TransformersService(EmbeddingService):
    """基于transformers的向量化服务"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers未安装，请运行: pip install transformers torch")
    
    async def load_model(self) -> bool:
        """加载transformers模型"""
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModel.from_pretrained(self.config.model_name)
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            
            logger.info(f"成功加载transformers模型: {self.config.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"加载transformers模型失败: {e}")
            return False
    
    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        if not self._model or not self._tokenizer:
            raise RuntimeError("模型未加载")
        
        try:
            embeddings = []
            
            # 分批处理
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                batch_embeddings = await self._encode_batch(batch_texts)
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"批量编码文本失败: {e}")
            return []
    
    async def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """编码一批文本"""
        def _encode():
            # Tokenize
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            if self.config.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self._model(**inputs)
                
                # 使用[CLS] token的embedding或者平均池化
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
                else:
                    # 平均池化
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                
                return embeddings.cpu().numpy().tolist()
        
        return await asyncio.get_event_loop().run_in_executor(None, _encode)
    
    async def encode_text(self, text: str) -> List[float]:
        """编码单个文本"""
        results = await self.encode_texts([text])
        return results[0] if results else []


class OpenAIEmbeddingService(EmbeddingService):
    """基于OpenAI API的向量化服务"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai未安装，请运行: pip install openai")
        
        if not config.openai_api_key:
            raise ValueError("OpenAI API密钥未提供")
        
        self._client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
    
    async def load_model(self) -> bool:
        """OpenAI API不需要本地加载模型"""
        try:
            # 测试API连接
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.embeddings.create,
                input="test",
                model=self.config.model_name
            )
            
            # 更新向量维度
            if response.data:
                self.config.vector_size = len(response.data[0].embedding)
            
            logger.info(f"成功连接OpenAI API，模型: {self.config.model_name}")
            logger.info(f"向量维度: {self.config.vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"连接OpenAI API失败: {e}")
            return False
    
    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本"""
        try:
            embeddings = []
            
            # OpenAI API有批量限制，分批处理
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._client.embeddings.create,
                    input=batch_texts,
                    model=self.config.model_name
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI批量编码失败: {e}")
            return []
    
    async def encode_text(self, text: str) -> List[float]:
        """编码单个文本"""
        results = await self.encode_texts([text])
        return results[0] if results else []


def create_embedding_service(config: EmbeddingConfig) -> EmbeddingService:
    """创建向量化服务实例"""
    if config.model_type == "sentence_transformers":
        return SentenceTransformersService(config)
    elif config.model_type == "transformers":
        return TransformersService(config)
    elif config.model_type == "openai":
        return OpenAIEmbeddingService(config)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")


# 预定义的模型配置
PRESET_MODELS = {
    "all-MiniLM-L6-v2": EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        model_type="sentence_transformers",
        vector_size=384
    ),
    "all-mpnet-base-v2": EmbeddingConfig(
        model_name="all-mpnet-base-v2",
        model_type="sentence_transformers",
        vector_size=768
    ),
    "text-embedding-ada-002": EmbeddingConfig(
        model_name="text-embedding-ada-002",
        model_type="openai",
        vector_size=1536
    ),
    "text-embedding-3-small": EmbeddingConfig(
        model_name="text-embedding-3-small",
        model_type="openai",
        vector_size=1536
    ),
    "text-embedding-3-large": EmbeddingConfig(
        model_name="text-embedding-3-large",
        model_type="openai",
        vector_size=3072
    )
}