"""
向量数据库抽象接口
支持多种向量数据库的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import uuid


@dataclass
class VectorDocument:
    """向量文档数据结构"""
    id: str
    content: str
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    document: VectorDocument
    score: float
    distance: Optional[float] = None


@dataclass
class VectorDatabaseConfig:
    """向量数据库配置"""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "ai_trend_documents"
    vector_size: int = 768
    distance_metric: str = "cosine"  # cosine, euclidean, dot
    api_key: Optional[str] = None
    timeout: int = 30
    
    # 额外配置参数
    extra_config: Optional[Dict[str, Any]] = None


class VectorDatabase(ABC):
    """向量数据库抽象基类"""
    
    def __init__(self, config: VectorDatabaseConfig):
        self.config = config
        self._client = None
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接到向量数据库"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """断开数据库连接"""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, vector_size: int, 
                              distance_metric: str = "cosine") -> bool:
        """创建集合"""
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass
    
    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        pass
    
    @abstractmethod
    async def insert_documents(self, documents: List[VectorDocument], 
                             collection_name: Optional[str] = None) -> bool:
        """插入文档"""
        pass
    
    @abstractmethod
    async def update_document(self, document: VectorDocument, 
                            collection_name: Optional[str] = None) -> bool:
        """更新文档"""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str], 
                             collection_name: Optional[str] = None) -> bool:
        """删除文档"""
        pass
    
    @abstractmethod
    async def search_similar(self, query_vector: List[float], 
                           top_k: int = 10, 
                           collection_name: Optional[str] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """向量相似度搜索"""
        pass
    
    @abstractmethod
    async def search_by_text(self, query_text: str, 
                           top_k: int = 10,
                           collection_name: Optional[str] = None,
                           filter_conditions: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """文本搜索（需要先向量化）"""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str, 
                         collection_name: Optional[str] = None) -> Optional[VectorDocument]:
        """根据ID获取文档"""
        pass
    
    @abstractmethod
    async def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """获取集合信息"""
        pass
    
    @abstractmethod
    async def count_documents(self, collection_name: Optional[str] = None) -> int:
        """统计文档数量"""
        pass
    
    def get_collection_name(self, collection_name: Optional[str] = None) -> str:
        """获取集合名称"""
        return collection_name or self.config.collection_name