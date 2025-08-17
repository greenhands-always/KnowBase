"""
知识图谱相关数据模型定义
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

from .enums import EntityType


class Entity(BaseModel):
    """知识图谱实体模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: EntityType
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Relationship(BaseModel):
    """知识图谱关系模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    type: str
    strength: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class VectorData(BaseModel):
    """向量数据模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    article_id: str
    content: str
    embedding: List[float]
    embedding_model: str
    content_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)