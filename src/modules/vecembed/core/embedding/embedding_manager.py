"""
嵌入管理器
统一管理向量化和向量数据库操作
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import uuid
from datetime import datetime

from .vector_database import VectorDatabase, VectorDocument, SearchResult, VectorDatabaseConfig
from .qdrant_database import QdrantDatabase
from .embedding_service import EmbeddingService, EmbeddingConfig, create_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class DocumentInput:
    """文档输入数据结构"""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingManagerConfig:
    """嵌入管理器配置"""
    # 向量数据库配置
    vector_db_config: VectorDatabaseConfig
    
    # 向量化服务配置
    embedding_config: EmbeddingConfig
    
    # 管理器配置
    auto_create_collection: bool = True
    chunk_size: int = 1000  # 文档分块大小
    overlap_size: int = 100  # 分块重叠大小
    batch_size: int = 32    # 批处理大小


class EmbeddingManager:
    """嵌入管理器 - 统一的向量化和存储接口"""
    
    def __init__(self, config: EmbeddingManagerConfig):
        self.config = config
        
        # 初始化向量数据库
        self.vector_db: VectorDatabase = QdrantDatabase(config.vector_db_config)
        
        # 初始化向量化服务
        self.embedding_service: EmbeddingService = create_embedding_service(config.embedding_config)
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """初始化管理器"""
        try:
            # 连接向量数据库
            if not await self.vector_db.connect():
                logger.error("向量数据库连接失败")
                return False
            
            # 加载向量化模型
            if not await self.embedding_service.load_model():
                logger.error("向量化模型加载失败")
                return False
            
            # 更新配置中的向量维度
            vector_size = await self.embedding_service.get_vector_size()
            self.config.vector_db_config.vector_size = vector_size
            
            # 自动创建集合
            if self.config.auto_create_collection:
                collection_name = self.config.vector_db_config.collection_name
                if not await self.vector_db.collection_exists(collection_name):
                    await self.vector_db.create_collection(
                        collection_name,
                        vector_size,
                        self.config.vector_db_config.distance_metric
                    )
            
            self._initialized = True
            logger.info("嵌入管理器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"嵌入管理器初始化失败: {e}")
            return False
    
    async def close(self):
        """关闭管理器"""
        if self.vector_db:
            await self.vector_db.disconnect()
        self._initialized = False
        logger.info("嵌入管理器已关闭")
    
    async def add_documents(self, documents: List[DocumentInput], 
                          collection_name: Optional[str] = None,
                          chunk_documents: bool = True) -> bool:
        """添加文档到向量数据库"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        try:
            # 处理文档分块
            processed_docs = []
            for doc in documents:
                if chunk_documents and len(doc.content) > self.config.chunk_size:
                    chunks = self._chunk_text(doc.content)
                    for i, chunk in enumerate(chunks):
                        chunk_doc = DocumentInput(
                            content=chunk,
                            metadata={
                                **doc.metadata,
                                "original_id": doc.id,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "created_at": datetime.now().isoformat()
                            },
                            id=f"{doc.id}_chunk_{i}"
                        )
                        processed_docs.append(chunk_doc)
                else:
                    # 添加时间戳
                    doc.metadata["created_at"] = datetime.now().isoformat()
                    processed_docs.append(doc)
            
            # 批量向量化
            texts = [doc.content for doc in processed_docs]
            vectors = await self.embedding_service.encode_texts(texts)
            
            if len(vectors) != len(processed_docs):
                logger.error("向量化结果数量与文档数量不匹配")
                return False
            
            # 创建向量文档
            vector_docs = []
            for doc, vector in zip(processed_docs, vectors):
                vector_doc = VectorDocument(
                    id=doc.id,
                    content=doc.content,
                    vector=vector,
                    metadata=doc.metadata
                )
                vector_docs.append(vector_doc)
            
            # 批量插入
            success = await self.vector_db.insert_documents(vector_docs, collection_name)
            
            if success:
                logger.info(f"成功添加 {len(processed_docs)} 个文档（原始文档数: {len(documents)}）")
            
            return success
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    async def add_document(self, document: DocumentInput, 
                         collection_name: Optional[str] = None,
                         chunk_document: bool = True) -> bool:
        """添加单个文档"""
        return await self.add_documents([document], collection_name, chunk_document)
    
    async def search_similar_documents(self, query: str, 
                                     top_k: int = 10,
                                     collection_name: Optional[str] = None,
                                     filter_conditions: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """搜索相似文档"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        try:
            # 向量化查询文本
            query_vector = await self.embedding_service.encode_text(query)
            
            if not query_vector:
                logger.error("查询文本向量化失败")
                return []
            
            # 执行向量搜索
            results = await self.vector_db.search_similar(
                query_vector, top_k, collection_name, filter_conditions
            )
            
            logger.info(f"搜索查询: '{query}', 返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    async def search_by_vector(self, query_vector: List[float], 
                             top_k: int = 10,
                             collection_name: Optional[str] = None,
                             filter_conditions: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """使用向量搜索"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        return await self.vector_db.search_similar(
            query_vector, top_k, collection_name, filter_conditions
        )
    
    async def get_document(self, document_id: str, 
                         collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        """获取文档"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        return await self.vector_db.get_document(document_id, collection_name)
    
    async def delete_documents(self, document_ids: List[str], 
                             collection_name: Optional[str] = None) -> bool:
        """删除文档"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        return await self.vector_db.delete_documents(document_ids, collection_name)
    
    async def delete_document(self, document_id: str, 
                            collection_name: Optional[str] = None) -> bool:
        """删除单个文档"""
        return await self.delete_documents([document_id], collection_name)
    
    async def update_document(self, document: DocumentInput, 
                            collection_name: Optional[str] = None) -> bool:
        """更新文档"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        try:
            # 向量化文档
            vector = await self.embedding_service.encode_text(document.content)
            
            if not vector:
                logger.error("文档向量化失败")
                return False
            
            # 添加更新时间戳
            document.metadata["updated_at"] = datetime.now().isoformat()
            
            # 创建向量文档
            vector_doc = VectorDocument(
                id=document.id,
                content=document.content,
                vector=vector,
                metadata=document.metadata
            )
            
            return await self.vector_db.update_document(vector_doc, collection_name)
            
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False
    
    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """获取集合信息"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        return await self.vector_db.get_collection_info(collection_name)
    
    async def count_documents(self, collection_name: Optional[str] = None) -> int:
        """统计文档数量"""
        if not self._initialized:
            raise RuntimeError("管理器未初始化")
        
        return await self.vector_db.count_documents(collection_name)
    
    def _chunk_text(self, text: str) -> List[str]:
        """文本分块"""
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            
            # 如果不是最后一块，尝试在句号或换行符处分割
            if end < len(text):
                # 寻找最近的句号或换行符
                for i in range(end, max(start + self.config.chunk_size - 100, start), -1):
                    if text[i] in '.。\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个开始位置（考虑重叠）
            start = max(start + 1, end - self.config.overlap_size)
        
        return chunks
    
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            "vector_db_config": asdict(self.config.vector_db_config),
            "embedding_config": asdict(self.config.embedding_config),
            "manager_config": {
                "auto_create_collection": self.config.auto_create_collection,
                "chunk_size": self.config.chunk_size,
                "overlap_size": self.config.overlap_size,
                "batch_size": self.config.batch_size
            }
        }


# 便捷函数
async def create_embedding_manager(
    vector_db_host: str = "localhost",
    vector_db_port: int = 6333,
    collection_name: str = "ai_trend_documents",
    embedding_model: str = "all-MiniLM-L6-v2",
    embedding_type: str = "sentence_transformers"
) -> EmbeddingManager:
    """创建嵌入管理器的便捷函数"""
    
    # 向量数据库配置
    vector_db_config = VectorDatabaseConfig(
        host=vector_db_host,
        port=vector_db_port,
        collection_name=collection_name
    )
    
    # 向量化配置
    embedding_config = EmbeddingConfig(
        model_name=embedding_model,
        model_type=embedding_type
    )
    
    # 管理器配置
    manager_config = EmbeddingManagerConfig(
        vector_db_config=vector_db_config,
        embedding_config=embedding_config
    )
    
    # 创建并初始化管理器
    manager = EmbeddingManager(manager_config)
    await manager.initialize()
    
    return manager