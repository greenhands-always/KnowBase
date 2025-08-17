"""
核心接口定义
为所有PKM Copilot模块提供标准化的接口规范
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Iterator
from datetime import datetime

from ..models import (
    Article, RawContent, ProcessingResult, Entity, Relationship,
    VectorData, FilterConfig, SummaryConfig, SearchQuery, SearchResult,
    Collection, ProcessingJob, DataSource
)


class DataCollectorInterface(ABC):
    """数据收集接口"""
    
    @abstractmethod
    async def collect(self, source: DataSource) -> AsyncIterator[RawContent]:
        """从数据源收集原始内容"""
        pass
    
    @abstractmethod
    async def validate_source(self, source: DataSource) -> bool:
        """验证数据源配置是否有效"""
        pass
    
    @abstractmethod
    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """获取数据源信息"""
        pass


class VectorizerInterface(ABC):
    """向量化接口"""
    
    @abstractmethod
    async def vectorize_text(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """将文本转换为向量"""
        pass
    
    @abstractmethod
    async def vectorize_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """批量文本向量化"""
        pass
    
    @abstractmethod
    async def get_similar_vectors(self, vector: List[float], limit: int = 10) -> List[VectorData]:
        """基于向量获取相似内容"""
        pass
    
    @abstractmethod
    async def search_by_text(self, query: str, limit: int = 10) -> List[VectorData]:
        """基于文本搜索相似内容"""
        pass


class ContentFilterInterface(ABC):
    """内容过滤接口"""
    
    @abstractmethod
    async def filter_content(self, articles: List[Article], config: FilterConfig) -> List[Article]:
        """根据配置过滤内容"""
        pass
    
    @abstractmethod
    async def calculate_quality_score(self, article: Article) -> float:
        """计算内容质量评分"""
        pass
    
    @abstractmethod
    async def calculate_relevance_score(self, article: Article, user_preferences: Dict[str, Any]) -> float:
        """计算内容相关性评分"""
        pass


class SummarizerInterface(ABC):
    """内容总结接口"""
    
    @abstractmethod
    async def summarize_article(self, article: Article, config: SummaryConfig) -> str:
        """总结单篇文章"""
        pass
    
    @abstractmethod
    async def summarize_batch(self, articles: List[Article], config: SummaryConfig) -> List[str]:
        """批量总结文章"""
        pass
    
    @abstractmethod
    async def generate_report(self, articles: List[Article], config: SummaryConfig) -> str:
        """生成综合报告"""
        pass
    
    @abstractmethod
    async def extract_concepts(self, article: Article) -> List[str]:
        """提取文章概念"""
        pass
    
    @abstractmethod
    async def extract_keywords(self, article: Article) -> List[str]:
        """提取文章关键词"""
        pass


class KnowledgeGraphInterface(ABC):
    """知识图谱接口"""
    
    @abstractmethod
    async def add_entity(self, entity: Entity) -> str:
        """添加实体"""
        pass
    
    @abstractmethod
    async def add_relationship(self, relationship: Relationship) -> str:
        """添加关系"""
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体信息"""
        pass
    
    @abstractmethod
    async def find_related_entities(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Entity]:
        """查找相关实体"""
        pass
    
    @abstractmethod
    async def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Entity]:
        """搜索实体"""
        pass
    
    @abstractmethod
    async def get_entity_path(self, source_id: str, target_id: str) -> List[Relationship]:
        """获取实体间的关联路径"""
        pass


class CollectionInterface(ABC):
    """收藏管理接口"""
    
    @abstractmethod
    async def create_collection(self, name: str, description: Optional[str] = None) -> str:
        """创建收藏集合"""
        pass
    
    @abstractmethod
    async def add_to_collection(self, collection_id: str, article_id: str) -> bool:
        """添加文章到集合"""
        pass
    
    @abstractmethod
    async def remove_from_collection(self, collection_id: str, article_id: str) -> bool:
        """从集合中移除文章"""
        pass
    
    @abstractmethod
    async def get_collection(self, collection_id: str) -> Optional[Collection]:
        """获取收藏集合"""
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[Collection]:
        """列出所有收藏集合"""
        pass
    
    @abstractmethod
    async def search_collections(self, query: str) -> List[Collection]:
        """搜索收藏集合"""
        pass


class SearchInterface(ABC):
    """搜索接口"""
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """执行搜索"""
        pass
    
    @abstractmethod
    async def semantic_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """语义搜索"""
        pass
    
    @abstractmethod
    async def keyword_search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """关键词搜索"""
        pass
    
    @abstractmethod
    async def advanced_search(self, query: SearchQuery) -> List[SearchResult]:
        """高级搜索"""
        pass


class ProcessingPipelineInterface(ABC):
    """处理管道接口"""
    
    @abstractmethod
    async def create_job(self, source_config: Dict[str, Any], processing_config: Dict[str, Any]) -> str:
        """创建处理任务"""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> ProcessingJob:
        """获取任务状态"""
        pass
    
    @abstractmethod
    async def list_jobs(self) -> List[ProcessingJob]:
        """列出所有任务"""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        pass
    
    @abstractmethod
    async def get_job_results(self, job_id: str) -> List[ProcessingResult]:
        """获取任务结果"""
        pass


class DataSourceManagerInterface(ABC):
    """数据源管理接口"""
    
    @abstractmethod
    async def add_source(self, source: DataSource) -> str:
        """添加数据源"""
        pass
    
    @abstractmethod
    async def update_source(self, source_id: str, updates: Dict[str, Any]) -> bool:
        """更新数据源"""
        pass
    
    @abstractmethod
    async def remove_source(self, source_id: str) -> bool:
        """移除数据源"""
        pass
    
    @abstractmethod
    async def get_source(self, source_id: str) -> Optional[DataSource]:
        """获取数据源"""
        pass
    
    @abstractmethod
    async def list_sources(self) -> List[DataSource]:
        """列出所有数据源"""
        pass
    
    @abstractmethod
    async def sync_source(self, source_id: str) -> bool:
        """同步数据源"""
        pass
    
    @abstractmethod
    async def get_sync_status(self, source_id: str) -> Dict[str, Any]:
        """获取同步状态"""
        pass


class ConfigManagerInterface(ABC):
    """配置管理接口"""
    
    @abstractmethod
    async def get_config(self, module: str, key: str) -> Any:
        """获取配置"""
        pass
    
    @abstractmethod
    async def set_config(self, module: str, key: str, value: Any) -> bool:
        """设置配置"""
        pass
    
    @abstractmethod
    async def get_module_config(self, module: str) -> Dict[str, Any]:
        """获取模块配置"""
        pass
    
    @abstractmethod
    async def update_module_config(self, module: str, config: Dict[str, Any]) -> bool:
        """更新模块配置"""
        pass
    
    @abstractmethod
    async def validate_config(self, module: str, config: Dict[str, Any]) -> bool:
        """验证配置"""
        pass


class HealthCheckInterface(ABC):
    """健康检查接口"""
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        pass


class MetricsInterface(ABC):
    """指标接口"""
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        pass
    
    @abstractmethod
    async def reset_metrics(self) -> bool:
        """重置指标"""
        pass