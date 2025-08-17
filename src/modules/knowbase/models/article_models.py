"""
文章相关数据模型定义
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

from .enums import ContentType, ProcessingStatus


class Article(BaseModel):
    """统一文章模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    content: str
    summary: Optional[str] = None
    source: str
    url: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    collected_at: datetime = Field(default_factory=datetime.now)
    
    # 内容分类
    content_type: ContentType = ContentType.ARTICLE
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    
    # 语义信息
    concepts: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    
    # 向量化信息
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # 质量评分
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # 处理状态
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: Optional[str] = None
    
    # 关联信息
    related_articles: List[str] = Field(default_factory=list)
    knowledge_graph_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Collection(BaseModel):
    """收藏模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    article_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_public: bool = False
    owner_id: Optional[str] = None