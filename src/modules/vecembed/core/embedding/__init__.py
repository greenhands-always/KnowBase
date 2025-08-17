"""
向量化SDK
支持多种向量数据库和向量化模型的统一接口
"""

from .vector_database import (
    VectorDatabase,
    VectorDatabaseConfig,
    VectorDocument,
    SearchResult
)

from .qdrant_database import QdrantDatabase

from .embedding_service import (
    EmbeddingService,
    EmbeddingConfig,
    SentenceTransformersService,
    TransformersService,
    OpenAIEmbeddingService,
    create_embedding_service,
    PRESET_MODELS
)

from .embedding_manager import (
    EmbeddingManager,
    EmbeddingManagerConfig,
    DocumentInput,
    create_embedding_manager
)

from .config import (
    EmbeddingSystemConfig,
    DEFAULT_CONFIG,
    DEV_CONFIG,
    PROD_CONFIG,
    OPENAI_CONFIG
)

__all__ = [
    # 向量数据库
    "VectorDatabase",
    "VectorDatabaseConfig", 
    "VectorDocument",
    "SearchResult",
    "QdrantDatabase",
    
    # 向量化服务
    "EmbeddingService",
    "EmbeddingConfig",
    "SentenceTransformersService",
    "TransformersService", 
    "OpenAIEmbeddingService",
    "create_embedding_service",
    "PRESET_MODELS",
    
    # 嵌入管理器
    "EmbeddingManager",
    "EmbeddingManagerConfig",
    "DocumentInput",
    "create_embedding_manager",
    
    # 配置
    "EmbeddingSystemConfig",
    "DEFAULT_CONFIG",
    "DEV_CONFIG", 
    "PROD_CONFIG",
    "OPENAI_CONFIG"
]

# 版本信息
__version__ = "1.0.0"