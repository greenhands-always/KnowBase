"""
KnowBase 数据模型
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from src.core.models import DataSource, DataSourceType


class RSSConfig(BaseModel):
    """RSS配置"""
    url: str
    update_interval: int = Field(default=3600)  # 秒
    max_items: int = Field(default=50)
    include_content: bool = Field(default=True)
    filters: List[str] = Field(default_factory=list)


class EmailConfig(BaseModel):
    """邮件配置"""
    imap_server: str
    username: str
    password: str
    folder: str = Field(default="INBOX")
    max_emails: int = Field(default=50)
    filters: List[str] = Field(default_factory=list)
    mark_read: bool = Field(default=False)


class WebCrawlerConfig(BaseModel):
    """网页爬虫配置"""
    url: str
    selector: str = Field(default="article")
    headers: Dict[str, str] = Field(default_factory=dict)
    cookies: Dict[str, str] = Field(default_factory=dict)
    max_pages: int = Field(default=10)
    delay: float = Field(default=1.0)
    user_agent: str = Field(default="PKM Copilot Bot")


class SourceValidationResult(BaseModel):
    """数据源验证结果"""
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    info: Dict[str, Any] = Field(default_factory=dict)


class SyncResult(BaseModel):
    """同步结果"""
    source_id: str
    source_name: str
    success: bool
    items_collected: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    sync_time: datetime = Field(default_factory=datetime.now)
    duration: float = Field(default=0.0)


class SourceStatistics(BaseModel):
    """数据源统计"""
    source_id: str
    total_items: int = Field(default=0)
    last_sync: Optional[datetime] = None
    sync_count: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    average_items_per_sync: float = Field(default=0.0)
    last_error: Optional[str] = None


class CollectionSchedule(BaseModel):
    """收集计划"""
    source_id: str
    schedule_type: str = Field(default="interval")  # interval, cron, manual
    interval_minutes: Optional[int] = None
    cron_expression: Optional[str] = None
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    is_active: bool = Field(default=True)


class SourceHealth(BaseModel):
    """数据源健康状态"""
    source_id: str
    status: str = Field(default="unknown")  # healthy, warning, error, unknown
    last_check: datetime = Field(default_factory=datetime.now)
    response_time: Optional[float] = None
    error_count: int = Field(default=0)
    last_error: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)


class DataSourceExtended(DataSource):
    """扩展数据源模型"""
    statistics: SourceStatistics = Field(default_factory=lambda: SourceStatistics(source_id=""))
    health: SourceHealth = Field(default_factory=lambda: SourceHealth(source_id=""))
    schedule: Optional[CollectionSchedule] = None
    config_type: str = Field(default="basic")  # basic, rss, email, crawler
    
    def __init__(self, **data):
        super().__init__(**data)
        self.statistics.source_id = self.id
        self.health.source_id = self.id
        if self.schedule:
            self.schedule.source_id = self.id