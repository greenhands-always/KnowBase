"""
Qdrant向量数据库实现
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

from .vector_database import VectorDatabase, VectorDatabaseConfig, VectorDocument, SearchResult

logger = logging.getLogger(__name__)


class QdrantDatabase(VectorDatabase):
    """Qdrant向量数据库实现"""
    
    def __init__(self, config: VectorDatabaseConfig):
        super().__init__(config)
        self._client: Optional[QdrantClient] = None
        self._distance_mapping = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
    
    async def connect(self) -> bool:
        """连接到Qdrant数据库"""
        try:
            # 根据不同模式创建客户端
            if self.config.host == ":memory:":
                # 内存模式
                self._client = QdrantClient(":memory:")
                logger.info("已连接到Qdrant（内存模式）")
            elif self.config.port is None and self.config.host not in [":memory:", "localhost", "127.0.0.1"]:
                # 文件存储模式（当port为None且host不是标准服务器地址时）
                self._client = QdrantClient(path=self.config.host)
                logger.info(f"已连接到Qdrant（文件存储模式）: {self.config.host}")
            else:
                # 服务器模式
                if getattr(self.config, 'https', False):
                    url = f"https://{self.config.host}:{self.config.port}"
                else:
                    url = f"http://{self.config.host}:{self.config.port}"
                
                self._client = QdrantClient(
                    url=url,
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
                logger.info(f"已连接到Qdrant服务器: {url}")
            
            # 测试连接
            await asyncio.get_event_loop().run_in_executor(
                None, self._client.get_collections
            )
            
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self._client.get_collections
            )
            logger.info(f"连接成功，现有集合数量: {len(collections.collections)}")
            
            return True
            
        except Exception as e:
            logger.error(f"连接Qdrant数据库失败: {e}")
            return False
    
    async def disconnect(self):
        """断开数据库连接"""
        if self._client:
            try:
                self._client.close()
                self._client = None
                logger.info("已断开Qdrant数据库连接")
            except Exception as e:
                logger.error(f"断开Qdrant连接时出错: {e}")
    
    async def create_collection(self, collection_name: str, vector_size: int, 
                              distance_metric: str = "cosine") -> bool:
        """创建集合"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        try:
            distance = self._distance_mapping.get(distance_metric, Distance.COSINE)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.create_collection,
                collection_name,
                VectorParams(size=vector_size, distance=distance)
            )
            
            logger.info(f"成功创建集合: {collection_name}")
            return True
            
        except ResponseHandlingException as e:
            if "already exists" in str(e).lower():
                logger.info(f"集合已存在: {collection_name}")
                return True
            logger.error(f"创建集合失败: {e}")
            return False
        except Exception as e:
            logger.error(f"创建集合时出错: {e}")
            return False
    
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.delete_collection,
                collection_name
            )
            
            logger.info(f"成功删除集合: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        try:
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self._client.get_collections
            )
            
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
            
        except Exception as e:
            logger.error(f"检查集合存在性时出错: {e}")
            return False
    
    async def insert_documents(self, documents: List[VectorDocument], 
                             collection_name: Optional[str] = None) -> bool:
        """插入文档"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        collection_name = self.get_collection_name(collection_name)
        
        try:
            points = []
            for doc in documents:
                if not doc.vector:
                    raise ValueError(f"文档 {doc.id} 缺少向量数据")
                
                payload = {
                    "content": doc.content,
                    **(doc.metadata or {})
                }
                
                point = PointStruct(
                    id=doc.id,
                    vector=doc.vector,
                    payload=payload
                )
                points.append(point)
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.upsert,
                collection_name,
                points
            )
            
            logger.info(f"成功插入 {len(documents)} 个文档到集合 {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"插入文档失败: {e}")
            return False
    
    async def update_document(self, document: VectorDocument, 
                            collection_name: Optional[str] = None) -> bool:
        """更新文档"""
        return await self.insert_documents([document], collection_name)
    
    async def delete_documents(self, document_ids: List[str], 
                             collection_name: Optional[str] = None) -> bool:
        """删除文档"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        collection_name = self.get_collection_name(collection_name)
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.delete,
                collection_name,
                document_ids
            )
            
            logger.info(f"成功删除 {len(document_ids)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    async def search_similar(self, query_vector: List[float], 
                           top_k: int = 10, 
                           collection_name: Optional[str] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """向量相似度搜索"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        collection_name = self.get_collection_name(collection_name)
        
        try:
            # 构建过滤条件
            query_filter = None
            if filter_conditions:
                query_filter = self._build_filter(filter_conditions)
            
            search_result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.search,
                collection_name,
                query_vector,
                query_filter,
                top_k,
                True  # with_payload
            )
            
            results = []
            for point in search_result:
                doc = VectorDocument(
                    id=str(point.id),
                    content=point.payload.get("content", ""),
                    vector=point.vector,
                    metadata={k: v for k, v in point.payload.items() if k != "content"}
                )
                
                result = SearchResult(
                    document=doc,
                    score=point.score,
                    distance=getattr(point, 'distance', None)
                )
                results.append(result)
            
            logger.info(f"搜索返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    async def search_by_text(self, query_text: str, 
                           top_k: int = 10,
                           collection_name: Optional[str] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """文本搜索（需要先向量化）"""
        # 这里需要向量化服务来将文本转换为向量
        # 暂时抛出异常，提示需要向量化服务
        raise NotImplementedError("文本搜索需要向量化服务支持，请使用 EmbeddingManager")
    
    async def get_document(self, document_id: str, 
                         collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        """根据ID获取文档"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        collection_name = self.get_collection_name(collection_name)
        
        try:
            points = await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.retrieve,
                collection_name,
                [document_id],
                True  # with_payload
            )
            
            if not points:
                return None
            
            point = points[0]
            doc = VectorDocument(
                id=str(point.id),
                content=point.payload.get("content", ""),
                vector=point.vector,
                metadata={k: v for k, v in point.payload.items() if k != "content"}
            )
            
            return doc
            
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            return None
    
    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """获取集合信息"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        collection_name = self.get_collection_name(collection_name)
        
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.get_collection,
                collection_name
            )
            
            return {
                "name": info.config.params.vectors.size,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance.name,
                "points_count": info.points_count,
                "status": info.status.name
            }
            
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}
    
    async def count_documents(self, collection_name: Optional[str] = None) -> int:
        """统计文档数量"""
        if not self._client:
            raise RuntimeError("数据库未连接")
        
        collection_name = self.get_collection_name(collection_name)
        
        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                self._client.get_collection,
                collection_name
            )
            
            return info.points_count
            
        except Exception as e:
            logger.error(f"统计文档数量失败: {e}")
            return 0
    
    def _build_filter(self, filter_conditions: Dict[str, Any]) -> Filter:
        """构建Qdrant过滤条件"""
        conditions = []
        
        for key, value in filter_conditions.items():
            if isinstance(value, (str, int, float, bool)):
                condition = FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                conditions.append(condition)
        
        return Filter(must=conditions) if conditions else None