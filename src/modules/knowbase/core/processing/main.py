#!/usr/bin/env python3
"""
文章处理工具主入口
提供命令行接口和简化的API来使用文章处理工具
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.processing import (
    ConceptExtractor, ArticleProcessor, ProcessingPipeline,
    PipelineBuilder, ConfigManager, ConfigTemplates, ResultManager
)
from src.processing.database_processor import DatabaseIntegratedProcessor, DatabaseProcessorFactory
from src.processing.database_config import DatabaseConfigTemplates
from src.processing.processing_pipeline import PipelineConfig
from src.infrastructure.utils import PathUtil, LLMUtil


class ArticleProcessingTool:
    """文章处理工具主类"""
    
    def __init__(self):
        """初始化文章处理工具"""
        self.config = None
        self.processor = None
        self.db_processor = None
        self.pipeline = None
        self.results = []
        self.config_manager = ConfigManager()
        self.result_manager = ResultManager()
    
    def process_with_config(self, config_name: str, input_path: Optional[str] = None, 
                          output_path: Optional[str] = None) -> str:
        """使用预定义配置处理文章"""
        
        # 加载配置
        try:
            config = self.config_manager.load_config(config_name)
        except FileNotFoundError:
            print(f"配置 '{config_name}' 不存在，使用默认配置")
            config = ConfigTemplates.get_standard_config()
        
        # 覆盖输入输出路径（如果提供）
        if input_path:
            config.input_path = input_path
        if output_path:
            config.output_path = output_path
        
        # 使用新的配置创建方法创建文章处理器
        article_processor = ArticleProcessor.create_from_processing_config(config)
        
        # 创建处理管道
        pipeline_config = PipelineConfig(
            input_type=config.input_type,
            input_path=config.input_path,
            file_pattern=config.file_pattern,
            min_file_size=config.min_file_size,
            max_files=config.max_files,
            output_path=config.output_path,
            output_format=config.output_format,
            enable_progress=config.enable_progress
        )
        
        pipeline = ProcessingPipeline(article_processor, pipeline_config)
        
        # 运行处理
        print(f"开始处理文章...")
        print(f"输入: {config.input_path}")
        print(f"输出: {config.output_path}")
        
        results = pipeline.run()
        
        # 打印统计信息
        pipeline.print_summary()
        
        # 保存摘要报告（如果启用）
        if config.save_summary_report:
            report_path = config.output_path.replace('.json', '_report.json')
            self.result_manager.save_summary_report(results, report_path)
            print(f"摘要报告已保存到: {report_path}")
        
        return config.output_path
    
    def quick_process(self, input_path: str, output_path: Optional[str] = None, 
                     processor_type: str = "standard") -> str:
        """快速处理文章"""
        
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = PathUtil.get_project_base_dir()
            output_path = PathUtil.concat_path(base_dir, f"result/quick_process_{timestamp}.json")
        
        # 创建LLM提供者
        provider = LLMUtil.OllamaProvider(model_name="zephyr")
        llm = provider.get_llm(model_name="zephyr")
        
        # 创建概念提取器
        concept_extractor = ConceptExtractor.create_llm_extractor(llm)
        
        # 创建文章处理器
        if processor_type == "basic":
            from src.processing.processors import BASIC_PROCESSORS
            article_processor = ArticleProcessor.create_custom_processor(
                concept_extractor, BASIC_PROCESSORS
            )
        elif processor_type == "full":
            from src.processing.processors import FULL_PROCESSORS
            article_processor = ArticleProcessor.create_custom_processor(
                concept_extractor, FULL_PROCESSORS
            )
        else:
            article_processor = ArticleProcessor.create_standard_processor(concept_extractor)
        
        # 使用构建器创建管道
        pipeline = (PipelineBuilder()
                   .with_processor(article_processor)
                   .with_input_directory(input_path, "*.md")
                   .with_output(output_path, "json")
                   .with_progress(True)
                   .build())
        
        # 运行处理
        print(f"快速处理文章...")
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        
        results = pipeline.run()
        pipeline.print_summary()
        
        return output_path
    
    def list_configs(self):
        """列出所有可用配置"""
        configs = self.config_manager.list_configs()
        
        if not configs:
            print("没有找到配置文件")
            return
        
        print("可用配置:")
        for config_name in configs:
            try:
                config = self.config_manager.load_config(config_name)
                print(f"  {config_name}:")
                print(f"    - LLM模型: {config.llm_model}")
                print(f"    - 处理器类型: {config.processor_type}")
                print(f"    - 输出格式: {config.output_format}")
            except Exception as e:
                print(f"  {config_name}: 加载失败 ({e})")
    
    def create_default_configs(self):
        """创建默认配置文件"""
        from src.processing.config import create_default_configs
        
        print("创建默认配置文件...")
        saved_configs = create_default_configs()
        
        print("已创建以下配置:")
        for name, file_path in saved_configs:
            print(f"  {name}: {file_path}")
    
    def create_database_configs(self):
        """创建数据库配置文件"""
        from .database_config import create_database_configs
        
        print("创建数据库配置文件...")
        saved_configs = create_database_configs()
        
        print("已创建以下数据库配置:")
        for name, file_path in saved_configs:
            print(f"  {name}: {file_path}")
    
    def process_with_database(self, config_name: str, input_path: Optional[str] = None, 
                            output_path: Optional[str] = None) -> str:
        """使用数据库配置处理文章"""
        
        # 加载配置
        try:
            config = self.config_manager.load_config(config_name)
        except FileNotFoundError:
            print(f"配置 '{config_name}' 不存在")
            return None
        
        # 确保启用数据库
        config.enable_database = True
        
        # 覆盖输入输出路径（如果提供）
        if input_path:
            config.input_path = input_path
        if output_path:
            config.output_path = output_path
        
        # 创建数据库集成处理器
        try:
            # 首先创建基础处理器
            base_processor = ArticleProcessor.create_from_processing_config(config)
            
            # 转换acquisition_method为枚举类型
            from ..infrastructure.utils.db.DatabaseManager import AcquisitionMethod
            if isinstance(config.acquisition_method, str):
                acquisition_method = AcquisitionMethod(config.acquisition_method)
            else:
                acquisition_method = config.acquisition_method
            
            # 然后创建数据库集成处理器
            db_processor = DatabaseProcessorFactory.create_processor(
                base_processor=base_processor,
                data_source_name=config.data_source_name,
                acquisition_method=acquisition_method
            )
        except Exception as e:
            print(f"创建数据库处理器失败: {e}")
            print("请确保数据库连接正常且表结构已创建")
            return None
        
        # 创建处理管道
        pipeline_config = PipelineConfig(
            input_type=config.input_type,
            input_path=config.input_path,
            file_pattern=config.file_pattern,
            min_file_size=config.min_file_size,
            max_files=config.max_files,
            output_path=config.output_path,
            output_format=config.output_format,
            enable_progress=config.enable_progress
        )
        
        pipeline = ProcessingPipeline(db_processor, pipeline_config)
        
        # 运行处理
        print(f"开始处理文章（数据库模式）...")
        print(f"输入: {config.input_path}")
        print(f"输出: {config.output_path}")
        print(f"数据源: {config.data_source_name}")
        
        results = pipeline.run()
        
        # 打印统计信息
        pipeline.print_summary()
        
        # 获取数据库处理统计
        try:
            db_stats = db_processor.get_processing_stats()
            print(f"\n数据库处理统计:")
            print(f"  - 总文章数: {db_stats.get('total_articles', 0)}")
            print(f"  - 成功处理: {db_stats.get('processed_articles', 0)}")
            print(f"  - 处理失败: {db_stats.get('failed_articles', 0)}")
            print(f"  - 重复文章: {db_stats.get('duplicate_articles', 0)}")
        except Exception as e:
            print(f"获取数据库统计失败: {e}")
        
        # 保存摘要报告（如果启用）
        if config.save_summary_report:
            report_path = config.output_path.replace('.json', '_report.json')
            self.result_manager.save_summary_report(results, report_path)
            print(f"摘要报告已保存到: {report_path}")
        
        return config.output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI趋势总结 - 文章处理工具")
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 使用配置处理
    config_parser = subparsers.add_parser('config', help='使用配置文件处理')
    config_parser.add_argument('config_name', help='配置名称')
    config_parser.add_argument('--input', help='输入路径（覆盖配置）')
    config_parser.add_argument('--output', help='输出路径（覆盖配置）')
    
    # 快速处理
    quick_parser = subparsers.add_parser('quick', help='快速处理')
    quick_parser.add_argument('input_path', help='输入路径')
    quick_parser.add_argument('--output', help='输出路径')
    quick_parser.add_argument('--type', choices=['basic', 'standard', 'full'], 
                             default='standard', help='处理器类型')
    
    # 列出配置
    subparsers.add_parser('list-configs', help='列出所有配置')
    
    # 创建默认配置
    subparsers.add_parser('init-configs', help='创建默认配置文件')
    
    # 创建数据库配置
    subparsers.add_parser('init-db-configs', help='创建数据库配置文件')
    
    # 数据库处理
    db_parser = subparsers.add_parser('database', help='使用数据库配置处理')
    db_parser.add_argument('config_name', help='数据库配置名称')
    db_parser.add_argument('--input', help='输入路径（覆盖配置）')
    db_parser.add_argument('--output', help='输出路径（覆盖配置）')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    tool = ArticleProcessingTool()
    
    try:
        if args.command == 'config':
            output_path = tool.process_with_config(
                args.config_name, args.input, args.output
            )
            print(f"\n处理完成！结果保存到: {output_path}")
            
        elif args.command == 'quick':
            output_path = tool.quick_process(
                args.input_path, args.output, args.type
            )
            print(f"\n处理完成！结果保存到: {output_path}")
            
        elif args.command == 'list-configs':
            tool.list_configs()
            
        elif args.command == 'init-configs':
            tool.create_default_configs()
            
        elif args.command == 'init-db-configs':
            tool.create_database_configs()
            
        elif args.command == 'database':
            output_path = tool.process_with_database(
                args.config_name, args.input, args.output
            )
            if output_path:
                print(f"\n数据库处理完成！结果保存到: {output_path}")
            else:
                print("\n数据库处理失败！")
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()