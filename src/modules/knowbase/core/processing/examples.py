"""
文章处理工具使用示例
演示如何使用工程化的文章处理工具类
"""

import os
from pathlib import Path
from datetime import datetime

# 导入工具类
from src.processing import (
    ConceptExtractor, ArticleProcessor, ProcessingPipeline, 
    PipelineBuilder, PipelineConfig, ResultManager, ResultFormat
)
from src.processing.processors import FULL_PROCESSORS, BASIC_PROCESSORS
from src.infrastructure.utils import PathUtil, LLMUtil


def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 1. 创建LLM提供者
    provider = LLMUtil.OllamaProvider(model_name="zephyr")
    llm = provider.get_llm(model_name="zephyr")
    
    # 2. 创建概念提取器
    concept_extractor = ConceptExtractor.create_llm_extractor(llm)
    
    # 3. 创建文章处理器
    article_processor = ArticleProcessor.create_standard_processor(concept_extractor)
    
    # 4. 处理单篇文章
    article_data = {
        "id": "test_article",
        "title": "AI技术发展趋势",
        "file_path": "d:/code/ai-trend-summary/result/Huggingface/Blog/Trending/10_Does_Present-Day_GenAI_Actually_Reason.md"
    }
    
    result = article_processor.process_article(article_data)
    
    print(f"处理结果:")
    print(f"- 标题: {result.title}")
    print(f"- 状态: {result.status}")
    print(f"- 概念数量: {len(result.concepts.concepts) if result.concepts else 0}")
    print(f"- 分类: {result.categories}")
    print(f"- 质量分: {result.quality_score}")


def example_pipeline_usage():
    """处理管道使用示例"""
    print("\n=== 处理管道使用示例 ===")
    
    # 1. 创建LLM提供者
    provider = LLMUtil.OllamaProvider(model_name="zephyr")
    llm = provider.get_llm(model_name="zephyr")
    
    # 2. 创建概念提取器
    concept_extractor = ConceptExtractor.create_llm_extractor(llm)
    
    # 3. 创建自定义文章处理器
    article_processor = ArticleProcessor.create_custom_processor(
        concept_extractor=concept_extractor,
        processors=FULL_PROCESSORS  # 使用完整的处理器集合
    )
    
    # 4. 使用构建器创建处理管道
    base_dir = PathUtil.get_project_base_dir()
    input_dir = PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")
    output_file = PathUtil.concat_path(base_dir, f"result/processed_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    pipeline = (PipelineBuilder()
                .with_processor(article_processor)
                .with_input_directory(input_dir, "*.md")
                .with_output(output_file, "json")
                .with_limits(max_files=3, min_file_size=100)  # 限制处理3个文件用于演示
                .build())
    
    # 5. 运行管道
    results = pipeline.run()
    
    # 6. 打印统计信息
    pipeline.print_summary()
    
    # 7. 生成摘要报告
    result_manager = ResultManager()
    report_file = PathUtil.concat_path(base_dir, f"result/processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    result_manager.save_summary_report(results, report_file)
    
    print(f"结果已保存到: {output_file}")
    print(f"报告已保存到: {report_file}")


def example_custom_processors():
    """自定义处理器示例"""
    print("\n=== 自定义处理器示例 ===")
    
    def custom_ai_scorer(result):
        """自定义AI相关性评分器"""
        if result.concepts:
            ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'deep learning', 'llm']
            all_text = ' '.join(result.concepts.concepts + result.concepts.keywords).lower()
            
            ai_score = sum(1 for keyword in ai_keywords if keyword in all_text) / len(ai_keywords)
            result.metadata['ai_relevance_score'] = ai_score
            
            if ai_score > 0.5:
                result.categories.append('High-AI-Relevance')
        
        return result
    
    def custom_length_analyzer(result):
        """自定义长度分析器"""
        if result.file_path and Path(result.file_path).exists():
            file_size = Path(result.file_path).stat().st_size
            
            if file_size > 10000:
                result.tags.append('Long-Form')
            elif file_size > 5000:
                result.tags.append('Medium-Form')
            else:
                result.tags.append('Short-Form')
            
            result.metadata['content_length_category'] = result.tags[-1]
        
        return result
    
    # 创建使用自定义处理器的处理器
    provider = LLMUtil.OllamaProvider(model_name="zephyr")
    llm = provider.get_llm(model_name="zephyr")
    concept_extractor = ConceptExtractor.create_llm_extractor(llm)
    
    custom_processors = BASIC_PROCESSORS + [custom_ai_scorer, custom_length_analyzer]
    
    article_processor = ArticleProcessor.create_custom_processor(
        concept_extractor=concept_extractor,
        processors=custom_processors
    )
    
    # 测试自定义处理器
    article_data = {
        "id": "custom_test",
        "title": "AI技术测试文章",
        "file_path": "d:/code/ai-trend-summary/result/Huggingface/Blog/Trending/10_Does_Present-Day_GenAI_Actually_Reason.md"
    }
    
    result = article_processor.process_article(article_data)
    
    print(f"自定义处理结果:")
    print(f"- AI相关性分数: {result.metadata.get('ai_relevance_score', 'N/A')}")
    print(f"- 内容长度分类: {result.metadata.get('content_length_category', 'N/A')}")
    print(f"- 标签: {result.tags}")


def example_result_management():
    """结果管理示例"""
    print("\n=== 结果管理示例 ===")
    
    # 创建一些示例结果
    from src.processing.article_processor import ProcessingResult, ProcessingStatus
    from src.processing.concept_extractor import ConceptExtractionResult
    
    sample_results = [
        ProcessingResult(
            article_id="article_1",
            title="AI发展趋势",
            status=ProcessingStatus.COMPLETED,
            concepts=ConceptExtractionResult(
                concepts=["人工智能", "机器学习", "深度学习"],
                entities=["OpenAI", "Google"],
                keywords=["AI", "ML", "技术"],
                confidence=0.9
            ),
            quality_score=0.85,
            importance_score=0.78,
            categories=["AI/ML"],
            tags=["AI", "技术", "趋势"]
        ),
        ProcessingResult(
            article_id="article_2", 
            title="编程最佳实践",
            status=ProcessingStatus.COMPLETED,
            concepts=ConceptExtractionResult(
                concepts=["编程", "最佳实践", "代码质量"],
                entities=["Python", "JavaScript"],
                keywords=["编程", "开发", "实践"],
                confidence=0.8
            ),
            quality_score=0.75,
            importance_score=0.70,
            categories=["Programming"],
            tags=["编程", "开发", "实践"]
        )
    ]
    
    # 创建结果管理器
    result_manager = ResultManager()
    
    # 保存为不同格式
    base_dir = PathUtil.get_project_base_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # JSON格式
    json_file = PathUtil.concat_path(base_dir, f"result/sample_results_{timestamp}.json")
    result_manager.save_results(sample_results, json_file, ResultFormat.JSON)
    print(f"JSON结果已保存到: {json_file}")
    
    # CSV格式
    csv_file = PathUtil.concat_path(base_dir, f"result/sample_results_{timestamp}.csv")
    result_manager.save_results(sample_results, csv_file, ResultFormat.CSV)
    print(f"CSV结果已保存到: {csv_file}")
    
    # 生成摘要报告
    summary = result_manager.generate_summary_report(sample_results)
    print(f"\n摘要报告:")
    for section, data in summary.items():
        print(f"{section}: {data}")


def example_configuration_based():
    """基于配置的使用示例"""
    print("\n=== 基于配置的使用示例 ===")
    
    # 创建配置
    base_dir = PathUtil.get_project_base_dir()
    
    config = PipelineConfig(
        input_type="directory",
        input_path=PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending"),
        file_pattern="*.md",
        min_file_size=100,
        max_files=2,  # 限制处理2个文件用于演示
        output_path=PathUtil.concat_path(base_dir, f"result/config_based_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
        output_format="json",
        enable_progress=True
    )
    
    # 创建处理器
    provider = LLMUtil.OllamaProvider(model_name="zephyr")
    llm = provider.get_llm(model_name="zephyr")
    concept_extractor = ConceptExtractor.create_llm_extractor(llm)
    article_processor = ArticleProcessor.create_standard_processor(concept_extractor)
    
    # 创建并运行管道
    pipeline = ProcessingPipeline(article_processor, config)
    results = pipeline.run()
    
    print(f"基于配置的处理完成，共处理 {len(results)} 篇文章")
    pipeline.print_summary()


if __name__ == "__main__":
    print("文章处理工具使用示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        example_basic_usage()
        example_pipeline_usage()
        example_custom_processors()
        example_result_management()
        example_configuration_based()
        
        print("\n所有示例运行完成！")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()