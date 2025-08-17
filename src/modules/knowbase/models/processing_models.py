"""
处理相关数据模型定义
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

from .enums import ProcessingStatus, DataSourceType


class RawContent(BaseModel):
    """原始内容模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: DataSourceType
    source_config: Dict[str, Any]
    raw_data: Dict[str, Any]
    collected_at: datetime = Field(default_factory=datetime.now)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING


class ProcessingResult(BaseModel):
    """处理结果模型"""
    article_id: str
    status: ProcessingStatus
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    concepts_extracted: List[str] = Field(default_factory=list)
    keywords_extracted: List[str] = Field(default_factory=list)
    summary_generated: Optional[str] = None
    embedding_generated: bool = False
    processed_at: datetime = Field(default_factory=datetime.now)


class ProcessingJob(BaseModel):
    """处理任务模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    source_config: Dict[str, Any]
    processing_config: Dict[str, Any]
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results_count: int = 0


class DataSource(BaseModel):
    """数据源模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: DataSourceType
    config: Dict[str, Any]
    is_active: bool = True
    last_sync: Optional[datetime] = None
    sync_interval: Optional[int] = None  # 分钟
    created_at: datetime = Field(default_factory=datetime.now)


class SyncStatus(BaseModel):
    """同步状态模型"""
    source_id: str
    last_sync: datetime
    next_sync: Optional[datetime] = None
    articles_synced: int = 0
    errors: List[str] = Field(default_factory=list)
    is_successful: bool = True