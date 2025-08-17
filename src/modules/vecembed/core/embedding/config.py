"""
向量化配置管理
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import json

from .embedding_manager import EmbeddingManagerConfig
from .vector_database import VectorDatabaseConfig
from .embedding_service import EmbeddingConfig


@dataclass
class EmbeddingSystemConfig:
    """向量化系统配置"""
    # 默认配置
    default_collection: str = "ai_trend_documents"
    
    # Qdrant配置
    qdrant_host: str = "localhost"
    qdrant_port: Optional[int] = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_https: bool = False
    qdrant_path: Optional[str] = None  # 文件存储模式路径
    
    # 默认向量化模型
    default_embedding_model: str = "all-MiniLM-L6-v2"
    default_embedding_type: str = "sentence_transformers"
    
    # OpenAI配置
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    
    # 处理配置
    chunk_size: int = 1000
    overlap_size: int = 100
    batch_size: int = 32
    
    # 设备配置
    device: str = "cpu"  # cpu, cuda
    
    @classmethod
    def from_env(cls) -> "EmbeddingSystemConfig":
        """从环境变量创建配置"""
        return cls(
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            default_embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            default_embedding_type=os.getenv("EMBEDDING_TYPE", "sentence_transformers"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            overlap_size=int(os.getenv("OVERLAP_SIZE", "100")),
            batch_size=int(os.getenv("BATCH_SIZE", "32")),
            device=os.getenv("DEVICE", "cpu")
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> "EmbeddingSystemConfig":
        """从配置文件创建配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def to_file(self, config_path: str):
        """保存配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
    
    def create_manager_config(self, collection_name: str) -> 'EmbeddingManagerConfig':
        """创建嵌入管理器配置"""
        from .embedding_manager import EmbeddingManagerConfig
        
        # 根据不同模式创建向量数据库配置
        if self.qdrant_host == ":memory:":
            # 内存模式
            vector_db_config = VectorDatabaseConfig(
                host=":memory:",
                port=0,  # 内存模式不需要端口
                collection_name=collection_name,
                api_key=self.qdrant_api_key
            )
        elif self.qdrant_path:
            # 文件存储模式
            vector_db_config = VectorDatabaseConfig(
                host=self.qdrant_path,
                port=0,  # 文件模式不需要端口
                collection_name=collection_name,
                api_key=self.qdrant_api_key
            )
        else:
            # 服务器模式
            vector_db_config = VectorDatabaseConfig(
                host=self.qdrant_host,
                port=self.qdrant_port or 6333,
                collection_name=collection_name,
                api_key=self.qdrant_api_key
            )
        
        embedding_config = EmbeddingConfig(
            model_name=self.default_embedding_model,
            model_type=self.default_embedding_type,
            device=self.device,
            openai_api_key=self.openai_api_key,
            openai_base_url=self.openai_base_url
        )
        
        return EmbeddingManagerConfig(
            vector_db_config=vector_db_config,
            embedding_config=embedding_config,
            chunk_size=self.chunk_size,
            overlap_size=self.overlap_size,
            batch_size=self.batch_size
        )


# 预定义配置模板
DEFAULT_CONFIG = EmbeddingSystemConfig()

# 开发配置 - 内存模式
DEV_CONFIG = EmbeddingSystemConfig(
    qdrant_host=":memory:",
    qdrant_port=None,
    default_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    default_embedding_type="sentence_transformers",
    chunk_size=500,
    overlap_size=50
)

# 文件存储配置
FILE_CONFIG = EmbeddingSystemConfig(
    qdrant_host="./qdrant_data",
    qdrant_port=None,
    qdrant_path="./qdrant_data",
    default_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    default_embedding_type="sentence_transformers"
)

# 生产配置
PROD_CONFIG = EmbeddingSystemConfig(
    qdrant_host="localhost",
    qdrant_port=6333,
    default_embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    default_embedding_type="sentence_transformers",
    batch_size=64,
    chunk_size=1000,
    overlap_size=200
)

# OpenAI配置
OPENAI_CONFIG = EmbeddingSystemConfig(
    qdrant_host="localhost",
    qdrant_port=6333,
    default_embedding_model="text-embedding-ada-002",
    default_embedding_type="openai",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_base_url="https://api.openai.com/v1"
)