"""
配置管理模块
提供统一的配置管理，支持模块间协作配置
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class DatabaseConfig(BaseModel):
    """数据库配置"""
    postgres_url: str = Field(default="postgresql://user:password@localhost/pkm_copilot")
    mongodb_url: str = Field(default="mongodb://localhost:27017/pkm_copilot")
    redis_url: str = Field(default="redis://localhost:6379")
    neo4j_url: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")


class VectorStoreConfig(BaseModel):
    """向量存储配置"""
    provider: str = Field(default="qdrant")  # qdrant, pinecone, weaviate
    url: str = Field(default="http://localhost:6333")
    api_key: Optional[str] = None
    collection_name: str = Field(default="pkm_articles")
    vector_size: int = Field(default=384)


class LLMConfig(BaseModel):
    """大语言模型配置"""
    provider: str = Field(default="ollama")  # openai, anthropic, ollama, local
    model_name: str = Field(default="llama2")
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)


class ModuleConfig(BaseModel):
    """模块配置基础类"""
    enabled: bool = Field(default=True)
    config_path: Optional[str] = None


class KnowBaseConfig(ModuleConfig):
    """KnowBase配置"""
    sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    sync_interval: int = Field(default=3600)  # 秒
    max_articles_per_source: int = Field(default=100)
    cleanup_old_articles: bool = Field(default=True)
    retention_days: int = Field(default=365)


class VecEmbedConfig(ModuleConfig):
    """VecEmbed配置"""
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    batch_size: int = Field(default=32)
    max_text_length: int = Field(default=512)
    use_gpu: bool = Field(default=False)
    cache_embeddings: bool = Field(default=True)


class SiftFlowConfig(ModuleConfig):
    """SiftFlow配置"""
    quality_threshold: float = Field(default=0.7)
    relevance_model: str = Field(default="default")
    filter_duplicates: bool = Field(default=True)
    max_duplicate_similarity: float = Field(default=0.9)


class SumAgentConfig(ModuleConfig):
    """SumAgent配置"""
    summary_model: str = Field(default="default")
    summary_length: int = Field(default=300)
    extract_concepts: bool = Field(default=True)
    extract_keywords: bool = Field(default=True)
    generate_titles: bool = Field(default=True)


class LinkVerseConfig(ModuleConfig):
    """LinkVerse配置"""
    graph_store: str = Field(default="neo4j")
    entity_extraction_model: str = Field(default="default")
    relationship_types: List[str] = Field(default_factory=lambda: [
        "related_to", "part_of", "mentions", "contradicts", "supports"
    ])
    max_relationship_distance: int = Field(default=3)


class CollectDeckConfig(ModuleConfig):
    """CollectDeck配置"""
    default_collections: List[str] = Field(default_factory=lambda: [
        "favorites", "reading_list", "research", "archive"
    ])
    enable_sharing: bool = Field(default=False)
    max_collection_size: int = Field(default=1000)
    enable_public_collections: bool = Field(default=False)


class PkmCopilotConfig(BaseModel):
    """PKM Copilot主配置"""
    
    # 基础配置
    app_name: str = Field(default="PKM Copilot")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # 数据库配置
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # 向量存储配置
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    # 大语言模型配置
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # 模块配置
    modules: Dict[str, ModuleConfig] = Field(default_factory=dict)
    
    # 具体模块配置
    knowbase: KnowBaseConfig = Field(default_factory=KnowBaseConfig)
    vecembed: VecEmbedConfig = Field(default_factory=VecEmbedConfig)
    siftflow: SiftFlowConfig = Field(default_factory=SiftFlowConfig)
    sumagent: SumAgentConfig = Field(default_factory=SumAgentConfig)
    linkverse: LinkVerseConfig = Field(default_factory=LinkVerseConfig)
    collectdeck: CollectDeckConfig = Field(default_factory=CollectDeckConfig)
    
    class Config:
        env_prefix = "PKM_"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self._config = None
        self._load_config()
    
    def _find_config_file(self) -> str:
        """查找配置文件"""
        possible_paths = [
            "pkm_copilot.yaml",
            "config/pkm_copilot.yaml",
            "~/.pkm_copilot/config.yaml",
            "/etc/pkm_copilot/config.yaml"
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        # 如果找不到，使用默认配置
        return "config/pkm_copilot.yaml"
    
    def _load_config(self):
        """加载配置"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        else:
            config_data = {}
        
        # 合并环境变量
        config_data = self._merge_env_vars(config_data)
        
        # 创建配置对象
        self._config = PkmCopilotConfig(**config_data)
    
    def _merge_env_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """合并环境变量"""
        env_mappings = {
            'PKM_DEBUG': ['debug'],
            'PKM_LOG_LEVEL': ['log_level'],
            'PKM_DATABASE_POSTGRES_URL': ['database', 'postgres_url'],
            'PKM_DATABASE_MONGODB_URL': ['database', 'mongodb_url'],
            'PKM_DATABASE_REDIS_URL': ['database', 'redis_url'],
            'PKM_VECTOR_STORE_PROVIDER': ['vector_store', 'provider'],
            'PKM_VECTOR_STORE_URL': ['vector_store', 'url'],
            'PKM_VECTOR_STORE_API_KEY': ['vector_store', 'api_key'],
            'PKM_LLM_PROVIDER': ['llm', 'provider'],
            'PKM_LLM_MODEL_NAME': ['llm', 'model_name'],
            'PKM_LLM_API_KEY': ['llm', 'api_key'],
            'PKM_LLM_BASE_URL': ['llm', 'base_url'],
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_data, path, value)
        
        return config_data
    
    def _set_nested_value(self, data: Dict[str, Any], path: List[str], value: Any):
        """设置嵌套值"""
        current = data
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def get_config(self) -> PkmCopilotConfig:
        """获取配置"""
        return self._config
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """获取模块配置"""
        if hasattr(self._config, module_name):
            return getattr(self._config, module_name).dict()
        return {}
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            # 创建新的配置对象
            new_config = PkmCopilotConfig(**{**self._config.dict(), **updates})
            self._config = new_config
            return True
        except Exception as e:
            print(f"更新配置失败: {e}")
            return False
    
    def save_config(self, path: Optional[str] = None) -> bool:
        """保存配置"""
        try:
            save_path = path or self.config_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(self._config.dict(), f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(self._config.dict(), f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def reload_config(self):
        """重新加载配置"""
        self._load_config()
    
    def get_database_url(self, db_type: str) -> str:
        """获取数据库URL"""
        db_config = self._config.database
        
        if db_type == "postgres":
            return db_config.postgres_url
        elif db_type == "mongodb":
            return db_config.mongodb_url
        elif db_type == "redis":
            return db_config.redis_url
        elif db_type == "neo4j":
            return db_config.neo4j_url
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")
    
    def is_module_enabled(self, module_name: str) -> bool:
        """检查模块是否启用"""
        module_config = self.get_module_config(module_name)
        return module_config.get('enabled', True)
    
    def get_llm_config(self) -> LLMConfig:
        """获取大语言模型配置"""
        return self._config.llm
    
    def get_vector_store_config(self) -> VectorStoreConfig:
        """获取向量存储配置"""
        return self._config.vector_store


# 全局配置实例
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> PkmCopilotConfig:
    """获取全局配置"""
    return get_config_manager().get_config()


def create_default_config(path: str = "config/pkm_copilot.yaml"):
    """创建默认配置文件"""
    config = PkmCopilotConfig()
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, allow_unicode=True)
    
    print(f"默认配置文件已创建: {path}")
    return path


# 配置模板
CONFIG_TEMPLATES = {
    "development": {
        "debug": True,
        "log_level": "DEBUG",
        "database": {
            "postgres_url": "postgresql://dev_user:dev_password@localhost/pkm_dev",
            "mongodb_url": "mongodb://localhost:27017/pkm_dev",
            "redis_url": "redis://localhost:6379/1",
        },
        "vector_store": {
            "provider": "qdrant",
            "url": "http://localhost:6333",
            "collection_name": "pkm_dev"
        }
    },
    "production": {
        "debug": False,
        "log_level": "INFO",
        "database": {
            "postgres_url": "postgresql://prod_user:prod_password@localhost/pkm_prod",
            "mongodb_url": "mongodb://localhost:27017/pkm_prod",
            "redis_url": "redis://localhost:6379/0",
        },
        "vector_store": {
            "provider": "pinecone",
            "collection_name": "pkm_prod"
        }
    },
    "test": {
        "debug": True,
        "log_level": "DEBUG",
        "database": {
            "postgres_url": "postgresql://test_user:test_password@localhost/pkm_test",
            "mongodb_url": "mongodb://localhost:27017/pkm_test",
            "redis_url": "redis://localhost:6379/15",
        },
        "vector_store": {
            "provider": "qdrant",
            "url": "http://localhost:6333",
            "collection_name": "pkm_test"
        }
    }
}


def create_config_template(template_name: str, output_path: str) -> bool:
    """创建配置模板"""
    if template_name not in CONFIG_TEMPLATES:
        print(f"不支持的模板: {template_name}")
        return False
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(CONFIG_TEMPLATES[template_name], f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置模板已创建: {output_path}")
        return True
    except Exception as e:
        print(f"创建配置模板失败: {e}")
        return False