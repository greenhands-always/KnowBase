"""
预定义处理器模块
提供常用的文章处理器实现
"""

from typing import Dict, Any, List
from .article_processor import ProcessingResult


def quality_scorer(result: ProcessingResult) -> ProcessingResult:
    """
    质量评分处理器
    基于概念提取结果计算文章质量分数
    """
    if result.concepts and result.concepts.concepts:
        concept_count = len(result.concepts.concepts)
        confidence = result.concepts.confidence
        
        # 质量分数计算逻辑：
        # - 概念数量权重：每个概念贡献0.1分
        # - 置信度权重：直接使用置信度
        # - 最终分数取平均值，最大值为1.0
        result.quality_score = min(1.0, (concept_count * 0.1 + confidence) / 2)
    else:
        result.quality_score = 0.1
    
    return result


def importance_scorer(result: ProcessingResult) -> ProcessingResult:
    """
    重要性评分处理器
    基于实体和关键词数量计算重要性分数
    """
    if result.concepts:
        entity_count = len(result.concepts.entities)
        keyword_count = len(result.concepts.keywords)
        
        # 重要性分数计算逻辑：
        # - 实体权重：每个实体贡献0.15分
        # - 关键词权重：每个关键词贡献0.1分
        # - 最终分数取平均值，最大值为1.0
        result.importance_score = min(1.0, (entity_count * 0.15 + keyword_count * 0.1) / 2)
    else:
        result.importance_score = 0.1
    
    return result


def trending_scorer(result: ProcessingResult) -> ProcessingResult:
    """
    热度评分处理器
    基于热门关键词和概念计算热度分数
    """
    if not result.concepts:
        result.trending_score = 0.1
        return result
    
    # 定义热门关键词
    trending_keywords = {
        'ai', 'artificial intelligence', 'machine learning', 'deep learning',
        'llm', 'gpt', 'chatgpt', 'openai', 'claude', 'gemini',
        'transformer', 'neural network', 'nlp', 'computer vision',
        'generative ai', 'agi', 'automation', 'robotics'
    }
    
    # 计算热度分数
    all_text = ' '.join(result.concepts.concepts + result.concepts.keywords).lower()
    
    trending_count = sum(1 for keyword in trending_keywords if keyword in all_text)
    total_concepts = len(result.concepts.concepts) + len(result.concepts.keywords)
    
    if total_concepts > 0:
        result.trending_score = min(1.0, trending_count / total_concepts * 2)
    else:
        result.trending_score = 0.1
    
    return result


def category_classifier(result: ProcessingResult) -> ProcessingResult:
    """
    分类处理器
    基于概念和关键词对文章进行分类
    """
    if not result.concepts or not result.concepts.concepts:
        result.categories.append('General')
        return result
    
    # 合并所有文本进行分类
    all_text = ' '.join(
        result.concepts.concepts + 
        result.concepts.entities + 
        result.concepts.keywords
    ).lower()
    
    # 定义分类规则
    classification_rules = {
        'AI/ML': [
            'ai', 'artificial intelligence', 'machine learning', 'deep learning',
            'neural network', 'llm', 'gpt', 'transformer', 'nlp'
        ],
        'Programming': [
            'programming', 'software', 'development', 'code', 'coding',
            'python', 'javascript', 'java', 'framework', 'library'
        ],
        'Data Science': [
            'data', 'database', 'analytics', 'statistics', 'visualization',
            'big data', 'data mining', 'data analysis'
        ],
        'Web Development': [
            'web', 'frontend', 'backend', 'html', 'css', 'react',
            'vue', 'angular', 'node.js', 'api'
        ],
        'DevOps': [
            'devops', 'docker', 'kubernetes', 'ci/cd', 'deployment',
            'cloud', 'aws', 'azure', 'infrastructure'
        ],
        'Security': [
            'security', 'cybersecurity', 'encryption', 'authentication',
            'vulnerability', 'privacy', 'blockchain'
        ],
        'Mobile': [
            'mobile', 'android', 'ios', 'app', 'react native',
            'flutter', 'swift', 'kotlin'
        ],
        'Research': [
            'research', 'paper', 'study', 'experiment', 'analysis',
            'methodology', 'academic', 'publication'
        ]
    }
    
    # 应用分类规则
    for category, keywords in classification_rules.items():
        if any(keyword in all_text for keyword in keywords):
            result.categories.append(category)
    
    # 如果没有匹配到任何分类，设为General
    if not result.categories:
        result.categories.append('General')
    
    return result


def tag_generator(result: ProcessingResult) -> ProcessingResult:
    """
    标签生成处理器
    从概念和关键词中生成文章标签
    """
    if not result.concepts:
        return result
    
    # 清空现有标签
    result.tags.clear()
    
    # 从关键词中选择前5个作为标签
    if result.concepts.keywords:
        result.tags.extend(result.concepts.keywords[:5])
    
    # 如果关键词不足5个，从概念中补充
    if len(result.tags) < 5 and result.concepts.concepts:
        remaining_slots = 5 - len(result.tags)
        additional_tags = result.concepts.concepts[:remaining_slots]
        result.tags.extend(additional_tags)
    
    # 去重并保持顺序
    seen = set()
    unique_tags = []
    for tag in result.tags:
        if tag.lower() not in seen:
            seen.add(tag.lower())
            unique_tags.append(tag)
    
    result.tags = unique_tags
    
    return result


def summary_generator(result: ProcessingResult) -> ProcessingResult:
    """
    摘要生成处理器
    基于概念和分类生成简单的文章摘要
    """
    if not result.concepts:
        result.summary = f"文章《{result.title}》暂无可用摘要。"
        return result
    
    # 构建摘要
    summary_parts = [f"文章《{result.title}》"]
    
    # 添加主要概念
    if result.concepts.concepts:
        main_concepts = result.concepts.concepts[:3]
        summary_parts.append(f"主要涉及{', '.join(main_concepts)}等概念")
    
    # 添加分类信息
    if result.categories:
        summary_parts.append(f"属于{', '.join(result.categories)}领域")
    
    # 添加质量评价
    if result.quality_score:
        if result.quality_score >= 0.8:
            quality_desc = "高质量"
        elif result.quality_score >= 0.6:
            quality_desc = "中等质量"
        else:
            quality_desc = "基础质量"
        summary_parts.append(f"为{quality_desc}内容")
    
    result.summary = "，".join(summary_parts) + "。"
    
    return result


def metadata_enricher(result: ProcessingResult) -> ProcessingResult:
    """
    元数据丰富处理器
    添加额外的元数据信息
    """
    # 添加处理统计信息
    if result.concepts:
        result.metadata.update({
            "concept_count": len(result.concepts.concepts),
            "entity_count": len(result.concepts.entities),
            "keyword_count": len(result.concepts.keywords),
            "extraction_confidence": result.concepts.confidence
        })
    
    # 添加分类统计
    result.metadata.update({
        "category_count": len(result.categories),
        "tag_count": len(result.tags),
        "has_summary": bool(result.summary)
    })
    
    # 添加质量指标
    scores = {}
    if result.quality_score is not None:
        scores["quality"] = result.quality_score
    if result.importance_score is not None:
        scores["importance"] = result.importance_score
    if result.trending_score is not None:
        scores["trending"] = result.trending_score
    
    if scores:
        result.metadata["scores"] = scores
    
    return result


# 预定义的处理器组合
BASIC_PROCESSORS = [
    quality_scorer,
    importance_scorer,
    category_classifier,
    tag_generator
]

FULL_PROCESSORS = [
    quality_scorer,
    importance_scorer,
    trending_scorer,
    category_classifier,
    tag_generator,
    summary_generator,
    metadata_enricher
]

ANALYSIS_PROCESSORS = [
    quality_scorer,
    importance_scorer,
    trending_scorer,
    category_classifier,
    metadata_enricher
]


def get_processor_by_name(name: str):
    """
    根据名称获取处理器函数
    
    Args:
        name: 处理器名称
        
    Returns:
        处理器函数
        
    Raises:
        ValueError: 如果处理器名称不存在
    """
    processors = {
        'quality_scorer': quality_scorer,
        'importance_scorer': importance_scorer,
        'trending_scorer': trending_scorer,
        'category_classifier': category_classifier,
        'tag_generator': tag_generator,
        'summary_generator': summary_generator,
        'metadata_enricher': metadata_enricher
    }
    
    if name not in processors:
        available = ', '.join(processors.keys())
        raise ValueError(f"Unknown processor '{name}'. Available processors: {available}")
    
    return processors[name]