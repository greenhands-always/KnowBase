"""
VecEmbed 核心实现
多模态信息向量化引擎的核心逻辑
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
import numpy as np
from datetime import datetime

from src.core.models import VectorData
from src.core.interfaces import VectorizerInterface
from src.core.config import get_config

logger = logging.getLogger(__name__)


class SentenceTransformerVectorizer(VectorizerInterface):
    """Sentence Transformers向量化实现"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.vector_size = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    async def vectorize_text(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """将文本转换为向量"""
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.model.encode, text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"向量化文本失败: {e}")
            raise
    
    async def vectorize_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """批量文本向量化"""
        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self.model.encode, texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"批量向量化文本失败: {e}")
            raise
    
    async def get_similar_vectors(self, vector: List[float], limit: int = 10) -> List[VectorData]:
        """基于向量获取相似内容"""
        # 这里应该连接到向量数据库
        # 目前返回空列表
        return []
    
    async def search_by_text(self, query: str, limit: int = 10) -> List[VectorData]:
        """基于文本搜索相似内容"""
        # 这里应该连接到向量数据库
        # 目前返回空列表
        return []


class OpenAIEmbeddingsVectorizer(VectorizerInterface):
    """OpenAI Embeddings向量化实现"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")
    
    async def vectorize_text(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """将文本转换为向量"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.client.embeddings.create(
                    input=text,
                    model=model_name or self.model_name
                )
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI向量化文本失败: {e}")
            raise
    
    async def vectorize_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """批量文本向量化"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.embeddings.create(
                    input=texts,
                    model=model_name or self.model_name
                )
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"OpenAI批量向量化文本失败: {e}")
            raise
    
    async def get_similar_vectors(self, vector: List[float], limit: int = 10) -> List[VectorData]:
        """基于向量获取相似内容"""
        return []
    
    async def search_by_text(self, query: str, limit: int = 10) -> List[VectorData]:
        """基于文本搜索相似内容"""
        return []


class QdrantVectorStore:
    """Qdrant向量存储实现"""
    
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "pkm_articles"):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=url)
            self.collection_name = collection_name
            self.url = url
        except ImportError:
            raise ImportError("qdrant-client is required. Install with: pip install qdrant-client")
    
    async def create_collection(self, vector_size: int, distance: str = "Cosine") -> bool:
        """创建向量集合"""
        try:
            from qdrant_client.models import Distance, VectorParams
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=getattr(Distance, distance))
            )
            return True
        except Exception as e:
            logger.error(f"创建Qdrant集合失败: {e}")
            return False
    
    async def upsert_vectors(self, vectors: List[VectorData]) -> bool:
        """插入或更新向量"""
        try:
            from qdrant_client.models import PointStruct
            
            points = [
                PointStruct(
                    id=vector.id,
                    vector=vector.embedding,
                    payload={
                        "article_id": vector.article_id,
                        "content": vector.content,
                        "content_type": vector.content_type,
                        "metadata": vector.metadata,
                        "created_at": vector.created_at.isoformat()
                    }
                )
                for vector in vectors
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except Exception as e:
            logger.error(f"插入Qdrant向量失败: {e}")
            return False
    
    async def search_similar(self, query_vector: List[float], limit: int = 10) -> List[VectorData]:
        """搜索相似向量"""
        try:
            from qdrant_client.models import SearchRequest
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            vectors = []
            for result in search_result:
                payload = result.payload
                vector = VectorData(
                    id=result.id,
                    article_id=payload.get("article_id"),
                    content=payload.get("content"),
                    embedding=result.vector,
                    embedding_model="qdrant",
                    content_type=payload.get("content_type"),
                    metadata=payload.get("metadata", {}),
                    created_at=datetime.fromisoformat(payload.get("created_at"))
                )
                vectors.append(vector)
            
            return vectors
        except Exception as e:
            logger.error(f"Qdrant搜索失败: {e}")
            return []
    
    async def delete_by_article_id(self, article_id: str) -> bool:
        """根据文章ID删除向量"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "must": [{"key": "article_id", "match": {"value": article_id}}]
                }
            )
            return True
        except Exception as e:
            logger.error(f"删除Qdrant向量失败: {e}")
            return False


class VecEmbedCore:
    """VecEmbed核心实现"""
    
    def __init__(self):
        self.config = get_config().vecembed
        self.vectorizers = {}
        self.vector_store = None
        self._setup_vectorizers()
        self._setup_vector_store()
    
    def _setup_vectorizers(self):
        """设置向量化器"""
        try:
            # 根据配置选择向量化器
            if self.config.embedding_model.startswith("text-embedding"):
                # OpenAI模型
                llm_config = get_config().llm
                self.vectorizers['openai'] = OpenAIEmbeddingsVectorizer(
                    api_key=llm_config.api_key,
                    model_name=self.config.embedding_model
                )
            else:
                # Sentence Transformers模型
                self.vectorizers['sentence_transformers'] = SentenceTransformerVectorizer(
                    model_name=self.config.embedding_model
                )
        except Exception as e:
            logger.warning(f"设置向量化器失败，使用默认: {e}")
            self.vectorizers['sentence_transformers'] = SentenceTransformerVectorizer()
    
    def _setup_vector_store(self):
        """设置向量存储"""
        vector_config = get_config().vector_store
        
        if vector_config.provider == "qdrant":
            self.vector_store = QdrantVectorStore(
                url=vector_config.url,
                collection_name=vector_config.collection_name
            )
        else:
            logger.warning(f"不支持的向量存储: {vector_config.provider}")
    
    async def vectorize_text(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """将文本转换为向量"""
        if not text.strip():
            return []
        
        # 选择合适的向量化器
        vectorizer_name = self._select_vectorizer(model_name)
        vectorizer = self.vectorizers[vectorizer_name]
        
        return await vectorizer.vectorize_text(text, model_name)
    
    async def vectorize_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """批量文本向量化"""
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            return [[] for _ in texts]
        
        # 分批处理
        batch_size = self.config.batch_size
        results = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            vectorizer_name = self._select_vectorizer(model_name)
            vectorizer = self.vectorizers[vectorizer_name]
            
            batch_results = await vectorizer.vectorize_batch(batch, model_name)
            results.extend(batch_results)
        
        # 为原始空文本保持顺序
        final_results = []
        valid_idx = 0
        for text in texts:
            if text.strip():
                final_results.append(results[valid_idx])
                valid_idx += 1
            else:
                final_results.append([])
        
        return final_results
    
    async def store_vectors(self, vectors: List[VectorData]) -> bool:
        """存储向量"""
        if not self.vector_store:
            logger.error("向量存储未初始化")
            return False
        
        if not vectors:
            return True
        
        return await self.vector_store.upsert_vectors(vectors)
    
    async def search_similar(self, query: str, limit: int = 10) -> List[VectorData]:
        """搜索相似内容"""
        if not self.vector_store:
            logger.error("向量存储未初始化")
            return []
        
        # 向量化查询
        query_vector = await self.vectorize_text(query)
        if not query_vector:
            return []
        
        return await self.vector_store.search_similar(query_vector, limit)
    
    async def search_by_vector(self, vector: List[float], limit: int = 10) -> List[VectorData]:
        """基于向量搜索"""
        if not self.vector_store:
            logger.error("向量存储未初始化")
            return []
        
        return await self.vector_store.search_similar(vector, limit)
    
    async def delete_article_vectors(self, article_id: str) -> bool:
        """删除文章的向量"""
        if not self.vector_store:
            logger.error("向量存储未初始化")
            return False
        
        return await self.vector_store.delete_by_article_id(article_id)
    
    async def create_collection(self, vector_size: int) -> bool:
        """创建向量集合"""
        if not self.vector_store:
            logger.error("向量存储未初始化")
            return False
        
        return await self.vector_store.create_collection(vector_size)
    
    def _select_vectorizer(self, model_name: Optional[str] = None) -> str:
        """选择向量化器"""
        if model_name:
            if model_name.startswith("text-embedding"):
                return 'openai'
            else:
                return 'sentence_transformers'
        else:
            # 根据配置选择默认向量化器
            return list(self.vectorizers.keys())[0]
    
    def get_vector_size(self, model_name: Optional[str] = None) -> int:
        """获取向量维度"""
        if model_name and model_name.startswith("text-embedding"):
            # OpenAI embedding dimensions
            if "-3-small" in model_name:
                return 1536
            elif "-3-large" in model_name:
                return 3072
            elif "-ada-002" in model_name:
                return 1536
            else:
                return 1536
        else:
            # 默认Sentence Transformers
            return self.config.vector_size
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用模型"""
        models = {
            "sentence_transformers": [
                "all-MiniLM-L6-v2",
                "all-MiniLM-L12-v2",
                "all-mpnet-base-v2",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ],
            "openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002"
            ]
        }
        
        # 只返回已初始化的向量化器支持的模型
        available = {}
        for provider in self.vectorizers.keys():
            if provider in models:
                available[provider] = models[provider]
        
        return available