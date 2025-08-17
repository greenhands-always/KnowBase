"""
统一数据模型导入
"""

# 从各个模块导入所有模型
from .enums import (
    ContentType, ProcessingStatus, DataSourceType, EntityType
)

from .article_models import (
    Article, Collection
)

from .processing_models import (
    RawContent, ProcessingResult, ProcessingJob, DataSource, SyncStatus
)

from .knowledge_graph_models import (
    Entity, Relationship, VectorData
)

from .search_models import (
    SearchQuery, SearchResult
)

from .config_models import (
    FilterConfig, SummaryConfig
)

from .huggingface_models import (
    HuggingFaceArticle
)

# 为了向后兼容，可以保留原来的导入方式
__all__ = [
    'ContentType', 'ProcessingStatus', 'DataSourceType', 'EntityType',
    'Article', 'Collection',
    'RawContent', 'ProcessingResult', 'ProcessingJob', 'DataSource', 'SyncStatus',
    'Entity', 'Relationship', 'VectorData',
    'SearchQuery', 'SearchResult',
    'FilterConfig', 'SummaryConfig',
    'HuggingFaceArticle'
]