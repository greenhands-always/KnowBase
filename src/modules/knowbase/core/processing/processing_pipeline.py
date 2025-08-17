"""
处理管道模块
提供可配置的文章处理流水线
"""

from typing import List, Dict, Any, Optional, Callable, Iterator, Union
from pathlib import Path
from pydantic import BaseModel, Field
from datetime import datetime
import json

from .article_processor import BaseArticleProcessor, ProcessingResult
from .concept_extractor import BaseConceptExtractor
from ..infrastructure.utils.StreamUtil import FileStreamBuilder, StreamItem


class PipelineConfig(BaseModel):
    """处理管道配置"""
    # 输入配置
    input_type: str = Field(description="输入类型: directory, json_with_files, stream", default="directory")
    input_path: str = Field(description="输入路径")
    file_pattern: str = Field(description="文件匹配模式", default="*.md")
    
    # 过滤配置
    min_file_size: int = Field(description="最小文件大小", default=100)
    max_files: Optional[int] = Field(description="最大处理文件数", default=None)
    
    # 输出配置
    output_path: str = Field(description="输出路径")
    output_format: str = Field(description="输出格式: json, csv, database", default="json")
    
    # 处理配置
    batch_size: int = Field(description="批处理大小", default=10)
    enable_progress: bool = Field(description="是否显示进度", default=True)
    
    # 其他配置
    metadata: Dict[str, Any] = Field(description="额外配置", default_factory=dict)


class ProcessingPipeline:
    """文章处理流水线"""
    
    def __init__(self, 
                 processor: BaseArticleProcessor,
                 config: PipelineConfig):
        """
        初始化处理管道
        
        Args:
            processor: 文章处理器
            config: 管道配置
        """
        self.processor = processor
        self.config = config
        self.results: List[ProcessingResult] = []
        self.statistics = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "start_time": None,
            "end_time": None,
            "processing_time": 0.0
        }
    
    def run(self) -> List[ProcessingResult]:
        """
        运行处理管道
        
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        self.statistics["start_time"] = datetime.now()
        
        try:
            # 1. 构建输入流
            input_stream = self._build_input_stream()
            
            # 2. 转换为文章数据
            articles = self._stream_to_articles(input_stream)
            
            # 3. 批量处理
            self.results = self._process_articles(articles)
            
            # 4. 保存结果
            self._save_results()
            
            # 5. 更新统计信息
            self._update_statistics()
            
        except Exception as e:
            print(f"管道执行失败: {e}")
            raise
        finally:
            self.statistics["end_time"] = datetime.now()
            if self.statistics["start_time"]:
                self.statistics["processing_time"] = (
                    self.statistics["end_time"] - self.statistics["start_time"]
                ).total_seconds()
        
        return self.results
    
    def _build_input_stream(self) -> Iterator[StreamItem]:
        """构建输入流"""
        input_path = Path(self.config.input_path)
        
        if self.config.input_type == "directory":
            builder = FileStreamBuilder(input_path)
            
            # 添加过滤器
            from ..infrastructure.utils.StreamUtil import file_exists_filter, file_size_filter
            builder.add_filter(file_exists_filter)
            builder.add_filter(file_size_filter(min_size=self.config.min_file_size))
            
            # 设置限制
            if self.config.max_files:
                builder.set_limit(self.config.max_files)
            
            return builder.build_from_directory(self.config.file_pattern)
            
        elif self.config.input_type == "json_with_files":
            # 假设输入路径是JSON文件，需要配置中指定文件目录
            json_file = input_path
            file_dir = Path(self.config.metadata.get("file_directory", input_path.parent))
            
            builder = FileStreamBuilder(file_dir)
            builder.add_filter(file_exists_filter)
            builder.add_filter(file_size_filter(min_size=self.config.min_file_size))
            
            if self.config.max_files:
                builder.set_limit(self.config.max_files)
            
            return builder.build_from_json_with_files(json_file, file_dir)
        
        else:
            raise ValueError(f"不支持的输入类型: {self.config.input_type}")
    
    def _stream_to_articles(self, stream: Iterator[StreamItem]) -> List[Dict[str, Any]]:
        """将流转换为文章数据"""
        articles = []
        
        for item in stream:
            article_data = {
                "id": item.id,
                "title": item.title,
                "file_path": str(item.file_path),
                "metadata": item.metadata
            }
            
            # 如果有JSON数据，合并到文章数据中
            if "json_data" in item.metadata:
                article_data.update(item.metadata["json_data"])
                article_data["metadata"] = item.metadata  # 保留原始元数据
            
            articles.append(article_data)
        
        return articles
    
    def _process_articles(self, articles: List[Dict[str, Any]]) -> List[ProcessingResult]:
        """处理文章列表"""
        if self.config.enable_progress:
            def progress_callback(idx: int, title: str):
                print(f"正在处理第 {idx + 1}/{len(articles)} 个文件: {title}")
        else:
            progress_callback = None
        
        return self.processor.process_batch(articles, progress_callback)
    
    def _save_results(self):
        """保存处理结果"""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "json":
            self._save_as_json(output_path)
        elif self.config.output_format == "csv":
            self._save_as_csv(output_path)
        elif self.config.output_format == "database":
            self._save_to_database(output_path)
        else:
            raise ValueError(f"不支持的输出格式: {self.config.output_format}")
    
    def _save_as_json(self, output_path: Path):
        """保存为JSON格式"""
        from .result_manager import ResultManager, ResultFormat
        
        manager = ResultManager()
        manager.save_results(self.results, output_path, ResultFormat.JSON)
    
    def _save_as_csv(self, output_path: Path):
        """保存为CSV格式"""
        from .result_manager import ResultManager, ResultFormat
        
        manager = ResultManager()
        manager.save_results(self.results, output_path, ResultFormat.CSV)
    
    def _save_to_database(self, config_path: Path):
        """保存到数据库"""
        # TODO: 实现数据库保存逻辑
        print(f"数据库保存功能待实现，配置文件: {config_path}")
    
    def _update_statistics(self):
        """更新统计信息"""
        self.statistics["total_processed"] = len(self.results)
        
        for result in self.results:
            if result.status.value == "completed":
                self.statistics["successful"] += 1
            elif result.status.value == "failed":
                self.statistics["failed"] += 1
            elif result.status.value == "skipped":
                self.statistics["skipped"] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return self.statistics.copy()
    
    def print_summary(self):
        """打印处理摘要"""
        stats = self.statistics
        print(f"\n处理完成！")
        print(f"总处理文件数: {stats['total_processed']}")
        print(f"成功: {stats['successful']}")
        print(f"失败: {stats['failed']}")
        print(f"跳过: {stats['skipped']}")
        print(f"总耗时: {stats['processing_time']:.2f}秒")
        
        if stats['total_processed'] > 0:
            success_rate = stats['successful'] / stats['total_processed'] * 100
            print(f"成功率: {success_rate:.1f}%")


class PipelineBuilder:
    """处理管道构建器"""
    
    def __init__(self):
        self.processor = None
        self.config = None
    
    def with_processor(self, processor: BaseArticleProcessor) -> 'PipelineBuilder':
        """设置处理器"""
        self.processor = processor
        return self
    
    def with_config(self, config: Union[PipelineConfig, Dict[str, Any]]) -> 'PipelineBuilder':
        """设置配置"""
        if isinstance(config, dict):
            self.config = PipelineConfig(**config)
        else:
            self.config = config
        return self
    
    def with_input_directory(self, directory: str, pattern: str = "*.md") -> 'PipelineBuilder':
        """设置输入目录"""
        if not self.config:
            self.config = PipelineConfig(input_path=directory)
        else:
            self.config.input_type = "directory"
            self.config.input_path = directory
            self.config.file_pattern = pattern
        return self
    
    def with_output(self, output_path: str, format: str = "json") -> 'PipelineBuilder':
        """设置输出"""
        if not self.config:
            self.config = PipelineConfig(input_path="", output_path=output_path)
        else:
            self.config.output_path = output_path
            self.config.output_format = format
        return self
    
    def with_limits(self, max_files: Optional[int] = None, min_file_size: int = 100) -> 'PipelineBuilder':
        """设置限制"""
        if self.config:
            self.config.max_files = max_files
            self.config.min_file_size = min_file_size
        return self
    
    def build(self) -> ProcessingPipeline:
        """构建管道"""
        if not self.processor:
            raise ValueError("必须设置处理器")
        if not self.config:
            raise ValueError("必须设置配置")
        
        return ProcessingPipeline(self.processor, self.config)