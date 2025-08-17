"""
文章处理器模块
提供统一的文章处理接口和结果管理
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from .concept_extractor import ConceptExtractionResult


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingResult(BaseModel):
    """文章处理结果数据模型"""
    article_id: str = Field(description="文章唯一标识")
    title: str = Field(description="文章标题")
    file_path: Optional[str] = Field(description="文件路径", default=None)
    status: ProcessingStatus = Field(description="处理状态", default=ProcessingStatus.PENDING)
    
    # 处理结果
    concepts: Optional[ConceptExtractionResult] = Field(description="概念提取结果", default=None)
    summary: Optional[str] = Field(description="文章摘要", default=None)
    tags: List[str] = Field(description="文章标签", default_factory=list)
    categories: List[str] = Field(description="文章分类", default_factory=list)
    
    # 评分和质量指标
    quality_score: Optional[float] = Field(description="质量评分", default=None)
    importance_score: Optional[float] = Field(description="重要性评分", default=None)
    trending_score: Optional[float] = Field(description="热度评分", default=None)
    
    # 元数据
    processing_time: Optional[float] = Field(description="处理耗时(秒)", default=None)
    processed_at: Optional[datetime] = Field(description="处理时间", default=None)
    metadata: Dict[str, Any] = Field(description="额外元数据", default_factory=dict)
    errors: List[str] = Field(description="处理错误信息", default_factory=list)


class BaseArticleProcessor(ABC):
    """文章处理器抽象基类"""
    
    @abstractmethod
    def process_article(self, article_data: Dict[str, Any]) -> ProcessingResult:
        """处理单篇文章"""
        pass
    
    @abstractmethod
    def process_batch(self, articles: List[Dict[str, Any]], 
                     progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """批量处理文章"""
        pass


class StandardArticleProcessor(BaseArticleProcessor):
    """标准文章处理器实现"""
    
    def __init__(self, concept_extractor=None, config: Optional[Dict[str, Any]] = None):
        """
        初始化文章处理器
        
        Args:
            concept_extractor: 概念提取器实例
            config: 处理配置
        """
        self.concept_extractor = concept_extractor
        self.config = config or {}
        self.processors = []  # 处理器链
        
    def add_processor(self, processor: Callable[[ProcessingResult], ProcessingResult]) -> 'StandardArticleProcessor':
        """添加处理器到处理链"""
        self.processors.append(processor)
        return self
    
    def process_article(self, article_data: Dict[str, Any]) -> ProcessingResult:
        """
        处理单篇文章
        
        Args:
            article_data: 文章数据，包含title, file_path等字段
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = datetime.now()
        
        # 创建初始处理结果
        result = ProcessingResult(
            article_id=article_data.get('id', article_data.get('title', 'unknown')),
            title=article_data.get('title', ''),
            file_path=article_data.get('file_path'),
            status=ProcessingStatus.PROCESSING,
            processed_at=start_time,
            metadata=article_data.copy()
        )
        
        try:
            # 1. 概念提取
            if self.concept_extractor and result.file_path:
                result.concepts = self.concept_extractor.extract_from_file(result.file_path)
                
                # 从概念提取结果中生成标签
                if result.concepts:
                    result.tags.extend(result.concepts.keywords[:5])  # 取前5个关键词作为标签
            
            # 2. 应用处理器链
            for processor in self.processors:
                try:
                    result = processor(result)
                except Exception as e:
                    result.errors.append(f"处理器 {processor.__name__} 执行失败: {str(e)}")
            
            # 3. 计算处理时间
            end_time = datetime.now()
            result.processing_time = (end_time - start_time).total_seconds()
            
            # 4. 设置最终状态
            if result.errors:
                result.status = ProcessingStatus.FAILED
            else:
                result.status = ProcessingStatus.COMPLETED
                
        except Exception as e:
            result.status = ProcessingStatus.FAILED
            result.errors.append(f"处理失败: {str(e)}")
            end_time = datetime.now()
            result.processing_time = (end_time - start_time).total_seconds()
        
        return result
    
    def process_batch(self, articles: List[Dict[str, Any]], 
                     progress_callback: Optional[Callable[[int, str], None]] = None) -> List[ProcessingResult]:
        """
        批量处理文章
        
        Args:
            articles: 文章数据列表
            progress_callback: 进度回调函数
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        results = []
        
        for idx, article_data in enumerate(articles):
            if progress_callback:
                progress_callback(idx, article_data.get('title', f'Article {idx}'))
            
            result = self.process_article(article_data)
            results.append(result)
        
        return results


# 预定义的处理器函数
def quality_scorer(result: ProcessingResult) -> ProcessingResult:
    """质量评分处理器"""
    if result.concepts and result.concepts.concepts:
        # 基于概念数量和置信度计算质量分
        concept_count = len(result.concepts.concepts)
        confidence = result.concepts.confidence
        result.quality_score = min(1.0, (concept_count * 0.1 + confidence) / 2)
    else:
        result.quality_score = 0.1
    
    return result


def importance_scorer(result: ProcessingResult) -> ProcessingResult:
    """重要性评分处理器"""
    if result.concepts:
        # 基于实体数量和关键词数量计算重要性
        entity_count = len(result.concepts.entities)
        keyword_count = len(result.concepts.keywords)
        result.importance_score = min(1.0, (entity_count * 0.15 + keyword_count * 0.1) / 2)
    else:
        result.importance_score = 0.1
    
    return result


def category_classifier(result: ProcessingResult) -> ProcessingResult:
    """分类处理器"""
    if result.concepts and result.concepts.concepts:
        # 基于概念进行简单分类
        concepts_text = ' '.join(result.concepts.concepts).lower()
        
        if any(term in concepts_text for term in ['ai', 'artificial intelligence', 'machine learning', 'llm']):
            result.categories.append('AI/ML')
        
        if any(term in concepts_text for term in ['programming', 'software', 'development', 'code']):
            result.categories.append('Programming')
        
        if any(term in concepts_text for term in ['data', 'database', 'analytics']):
            result.categories.append('Data')
        
        if not result.categories:
            result.categories.append('General')
    
    return result


class ArticleProcessor:
    """文章处理器工厂类"""
    
    @staticmethod
    def create_standard_processor(concept_extractor=None, 
                                config: Optional[Dict[str, Any]] = None) -> StandardArticleProcessor:
        """创建标准文章处理器"""
        processor = StandardArticleProcessor(concept_extractor, config)
        
        # 添加默认处理器
        processor.add_processor(quality_scorer)
        processor.add_processor(importance_scorer)
        processor.add_processor(category_classifier)
        
        return processor
    
    @staticmethod
    def create_custom_processor(concept_extractor=None,
                              processors: List[Callable] = None,
                              config: Optional[Dict[str, Any]] = None) -> StandardArticleProcessor:
        """创建自定义文章处理器"""
        processor = StandardArticleProcessor(concept_extractor, config)
        
        if processors:
            for proc in processors:
                processor.add_processor(proc)
        
        return processor
    
    @staticmethod
    def create_from_processing_config(processing_config) -> StandardArticleProcessor:
        """从处理配置创建文章处理器"""
        from .concept_extractor import ConceptExtractor
        
        # 创建概念提取器
        concept_extractor = None
        if processing_config.enable_concept_extraction:
            try:
                concept_extractor = ConceptExtractor.create_llm_from_processing_config(processing_config)
            except Exception as e:
                print(f"警告: 无法创建概念提取器: {e}")
        
        # 创建处理器
        processor = StandardArticleProcessor(concept_extractor, processing_config.__dict__)
        
        # 根据配置添加处理器
        if processing_config.enable_quality_scoring:
            processor.add_processor(quality_scorer)
        
        if processing_config.enable_importance_scoring:
            processor.add_processor(importance_scorer)
        
        if processing_config.enable_categorization:
            processor.add_processor(category_classifier)
        
        # 添加自定义处理器
        if processing_config.custom_processors:
            from .processors import get_processor_by_name
            for processor_name in processing_config.custom_processors:
                try:
                    custom_proc = get_processor_by_name(processor_name)
                    if custom_proc:
                        processor.add_processor(custom_proc)
                except Exception as e:
                    print(f"警告: 无法加载自定义处理器 {processor_name}: {e}")
        
        return processor