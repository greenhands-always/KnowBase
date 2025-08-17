"""
搜索相关数据模型定义
"""

from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .article_models import Article
from .config_models import FilterConfig


class SearchQuery(BaseModel):
    """搜索查询模型"""
    query: str
    filters: Optional[FilterConfig] = None
    limit: int = 10
    offset: int = 0
    search_type: str = "semantic"  # semantic, keyword, hybrid
    sort_by: str = "relevance"
    include_vectors: bool = False


class SearchResult(BaseModel):
    """搜索结果模型"""
    article: Article
    score: float
    highlights: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field(default_factory=list)