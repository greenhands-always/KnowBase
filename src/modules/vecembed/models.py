"""
VecEmbed 数据模型
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class EmbeddingProvider(str, Enum):
    """向量化提供商"""
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class VectorStoreType(str, Enum):
    """向量存储类型"""
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CHROMA = "chroma"
    LOCAL = "local"


class EmbeddingRequest(BaseModel):
    """向量化请求"""
    text: str
    model_name: Optional[str] = None
    provider: Optional[EmbeddingProvider] = None
    truncate: bool = Field(default=True)
    normalize: bool = Field(default=True)


class BatchEmbeddingRequest(BaseModel):
    """批量向量化请求"""
    texts: List[str]
    model_name: Optional[str] = None
    provider: Optional[EmbeddingProvider] = None
    truncate: bool = Field(default=True)
    normalize: bool = Field(default=True)
    batch_size: Optional[int] = None


class EmbeddingResponse(BaseModel):
    """向量化响应"""
    embedding: List[float]
    model_name: str
    provider: str
    vector_size: int
    processing_time: float
    created_at: datetime = Field(default_factory=datetime.now)


class BatchEmbeddingResponse(BaseModel):
    """批量向量化响应"""
    embeddings: List[List[float]]
    model_name: str
    provider: str
    vector_size: int
    processing_time: float
    batch_size: int
    created_at: datetime = Field(default_factory=datetime.now)


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Dict[str, Any] = Field(default_factory=dict)
    include_metadata: bool = Field(default=True)
    search_type: str = Field(default="similarity")  # similarity, mmr, hybrid


class SearchResponse(BaseModel):
    """搜索响应"""
    results: List[Dict[str, Any]]
    query: str
    total_count: int
    processing_time: float
    search_type: str
    created_at: datetime = Field(default_factory=datetime.now)


class VectorDataRequest(BaseModel):
    """向量数据请求"""
    article_id: str
    content: str
    content_type: str = Field(default="text")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    chunk_total: Optional[int] = None


class VectorStorageRequest(BaseModel):
    """向量存储请求"""
    vectors: List[VectorDataRequest]
    model_name: str
    collection_name: Optional[str] = None
    overwrite: bool = Field(default=False)


class CollectionCreateRequest(BaseModel):
    """创建集合请求"""
    name: str
    vector_size: int
    distance_metric: str = Field(default="cosine")  # cosine, dot, euclidean
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    """集合信息"""
    name: str
    vector_size: int
    distance_metric: str
    points_count: int
    created_at: datetime
    metadata: Dict[str, Any]


class ModelInfo(BaseModel):
    """模型信息"""
    name: str
    provider: str
    vector_size: int
    max_tokens: int
    supported_languages: List[str]
    description: str
    is_available: bool = Field(default=True)


class HealthStatus(BaseModel):
    """健康状态"""
    status: str = Field(default="healthy")
    provider: str
    model_name: str
    vector_store: str
    collections: int
    total_vectors: int
    uptime: float
    last_error: Optional[str] = None


class PerformanceMetrics(BaseModel):
    """性能指标"""
    total_requests: int = Field(default=0)
    successful_requests: int = Field(default=0)
    failed_requests: int = Field(default=0)
    average_processing_time: float = Field(default=0.0)
    total_vectors_processed: int = Field(default=0)
    cache_hits: int = Field(default=0)
    cache_misses: int = Field(default=0)
    last_updated: datetime = Field(default_factory=datetime.now)


class CacheConfig(BaseModel):
    """缓存配置"""
    enabled: bool = Field(default=True)
    ttl: int = Field(default=3600)  # 秒
    max_size: int = Field(default=1000)
    eviction_policy: str = Field(default="LRU")


class ProcessingOptions(BaseModel):
    """处理选项"""
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    min_chunk_size: int = Field(default=100)
    max_chunk_size: int = Field(default=2000)
    split_method: str = Field(default="sentence")  # sentence, paragraph, token
    language: str = Field(default="zh")
    preprocess: bool = Field(default=True)
    remove_stopwords: bool = Field(default=True)
    stemming: bool = Field(default=False)


class VectorQuery(BaseModel):
    """向量查询"""
    vector: List[float]
    k: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    include_vectors: bool = Field(default=False)
    include_payload: bool = Field(default=True)
    filters: Dict[str, Any] = Field(default_factory=dict)