"""
文章处理模块
提供文章概念提取、处理和分析的工具类
"""

# 导入核心组件
from .concept_extractor import (
    ConceptExtractionResult,
    BaseConceptExtractor,
    LLMConceptExtractor,
    ConceptExtractor
)

from .article_processor import (
    ProcessingStatus,
    ProcessingResult,
    BaseArticleProcessor,
    StandardArticleProcessor,
    ArticleProcessor
)

from .processing_pipeline import (
    PipelineConfig,
    ProcessingPipeline,
    PipelineBuilder
)

from .result_manager import (
    ResultFormat,
    ResultManager
)

from .config import (
    ProcessingConfig,
    ConfigManager,
    ConfigTemplates
)

# 导入预定义处理器
from . import processors

# 导出公共接口
__all__ = [
    # 概念提取
    'ConceptExtractionResult',
    'BaseConceptExtractor', 
    'LLMConceptExtractor',
    'ConceptExtractor',
    
    # 文章处理
    'ProcessingStatus',
    'ProcessingResult',
    'BaseArticleProcessor',
    'StandardArticleProcessor', 
    'ArticleProcessor',
    
    # 处理管道
    'PipelineConfig',
    'ProcessingPipeline',
    'PipelineBuilder',
    
    # 结果管理
    'ResultFormat',
    'ResultManager',
    
    # 配置管理
    'ProcessingConfig',
    'ConfigManager',
    'ConfigTemplates',
    
    # 处理器模块
    'processors'
]