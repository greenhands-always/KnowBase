"""
枚举类型定义
为所有PKM Copilot模块提供标准化的枚举类型
"""

from enum import Enum


class ContentType(str, Enum):
    """内容类型枚举"""
    ARTICLE = "article"
    EMAIL = "email"
    RSS = "rss"
    SOCIAL = "social"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DataSourceType(str, Enum):
    """数据源类型枚举"""
    RSS = "rss"
    EMAIL = "email"
    CRAWLER = "crawler"
    API = "api"
    MANUAL = "manual"
    IMPORT = "import"


class EntityType(str, Enum):
    """实体类型枚举"""
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"