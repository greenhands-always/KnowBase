"""
配置相关数据模型定义
"""

from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class FilterConfig(BaseModel):
    """过滤配置模型"""
    min_quality_score: Optional[float] = None
    min_relevance_score: Optional[float] = None
    required_tags: List[str] = Field(default_factory=list)
    excluded_tags: List[str] = Field(default_factory=list)
    date_range: Optional[Dict[str, datetime]] = None
    source_whitelist: List[str] = Field(default_factory=list)
    source_blacklist: List[str] = Field(default_factory=list)


class SummaryConfig(BaseModel):
    """总结配置模型"""
    summary_type: str = "brief"  # brief, detailed, bullet_points
    max_length: int = 500
    include_keywords: bool = True
    include_concepts: bool = True
    output_format: str = "markdown"  # markdown, json, html
    language: str = "zh"