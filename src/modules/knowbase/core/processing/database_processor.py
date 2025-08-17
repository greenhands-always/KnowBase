"""
数据库集成的文章处理器
将处理结果自动存储到数据库
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging

from .article_processor import BaseArticleProcessor, ProcessingResult, ProcessingStatus
from ..infrastructure.utils.db.DatabaseManager import DatabaseManager, AcquisitionMethod, ProcessingStage


class DatabaseIntegratedProcessor(BaseArticleProcessor):
    """集成数据库存储的文章处理器"""
    
    def __init__(self, base_processor: BaseArticleProcessor, 
                 database_manager: Optional[DatabaseManager] = None,
                 data_source_name: str = "default",
                 acquisition_method: AcquisitionMethod = AcquisitionMethod.USER_IMPORT):
        """
        初始化数据库集成处理器
        
        Args:
            base_processor: 基础文章处理器
            database_manager: 数据库管理器
            data_source_name: 数据源名称
            acquisition_method: 获取方式
        """
        self.base_processor = base_processor
        self.db_manager = database_manager or DatabaseManager()
        self.data_source_name = data_source_name
        self.acquisition_method = acquisition_method
        self.logger = logging.getLogger(__name__)
        
        # 确保数据源存在
        self._ensure_data_source()
    
    def _ensure_data_source(self):
        """确保数据源存在"""
        try:
            self.source_id = self.db_manager.create_data_source(
                name=self.data_source_name,
                source_type="processing",
                acquisition_method=self.acquisition_method,
                description=f"文章处理数据源 - {self.data_source_name}"
            )
            self.logger.info(f"数据源已准备: {self.data_source_name} (ID: {self.source_id})")
        except Exception as e:
            self.logger.error(f"创建数据源失败: {e}")
            self.source_id = None
    
    def process_article(self, article_data: Dict[str, Any]) -> ProcessingResult:
        """
        处理单篇文章并存储到数据库
        
        Args:
            article_data: 文章数据
            
        Returns:
            ProcessingResult: 处理结果
        """
        # 首先使用基础处理器处理文章
        result = self.base_processor.process_article(article_data)
        
        # 如果数据源准备好了，存储到数据库
        if self.source_id:
            try:
                # 创建文章记录
                article_id = self._create_article_record(article_data, result)
                
                if article_id:
                    # 保存处理结果
                    success = self.db_manager.save_processing_result(result, article_id)
                    
                    if success:
                        result.metadata['database_article_id'] = article_id
                        self.logger.info(f"文章已保存到数据库: {result.title} (ID: {article_id})")
                    else:
                        self.logger.error(f"保存处理结果失败: {result.title}")
                        result.errors.append("数据库保存失败")
                
            except Exception as e:
                self.logger.error(f"数据库操作失败: {e}")
                result.errors.append(f"数据库操作失败: {str(e)}")
        
        return result
    
    def _create_article_record(self, article_data: Dict[str, Any], 
                             result: ProcessingResult) -> Optional[int]:
        """
        创建文章记录
        
        Args:
            article_data: 原始文章数据
            result: 处理结果
            
        Returns:
            Optional[int]: 文章ID
        """
        try:
            # 读取文章内容（如果有文件路径）
            content = None
            if result.file_path:
                try:
                    with open(result.file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    self.logger.warning(f"无法读取文件内容: {result.file_path}, {e}")
            
            # 创建文章记录
            article_id = self.db_manager.create_article(
                title=result.title,
                source_id=self.source_id,
                acquisition_method=self.acquisition_method,
                url=article_data.get('url'),
                author=article_data.get('author'),
                published_at=article_data.get('published_at'),
                content=content,
                external_id=article_data.get('external_id', result.article_id),
                metadata={
                    'file_path': result.file_path,
                    'processing_metadata': result.metadata,
                    'original_data': article_data
                }
            )
            
            return article_id
            
        except Exception as e:
            self.logger.error(f"创建文章记录失败: {e}")
            return None
    
    def process_batch(self, articles: List[Dict[str, Any]], 
                     progress_callback: Optional[Callable[[int, str], None]] = None) -> List[ProcessingResult]:
        """
        批量处理文章并存储到数据库
        
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
        
        # 记录批量处理统计
        self._log_batch_statistics(results)
        
        return results
    
    def _log_batch_statistics(self, results: List[ProcessingResult]):
        """记录批量处理统计信息"""
        total = len(results)
        completed = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
        
        self.logger.info(f"批量处理完成: 总计 {total}, 成功 {completed}, 失败 {failed}")
        
        # 获取数据库统计信息
        try:
            stats = self.db_manager.get_processing_statistics()
            if stats:
                self.logger.info(f"数据库统计: {stats['overall']}")
        except Exception as e:
            self.logger.warning(f"获取数据库统计失败: {e}")
    
    def get_article_status(self, article_id: int) -> Dict[str, Any]:
        """
        获取文章处理状态
        
        Args:
            article_id: 文章ID
            
        Returns:
            Dict: 处理状态信息
        """
        return self.db_manager.get_article_processing_status(article_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict: 统计信息
        """
        return self.db_manager.get_processing_statistics()
    
    def close(self):
        """关闭数据库连接"""
        if self.db_manager:
            self.db_manager.close()


class DatabaseProcessorFactory:
    """数据库处理器工厂"""
    
    @staticmethod
    def create_processor(base_processor: BaseArticleProcessor,
                        data_source_name: str = "file_processing",
                        acquisition_method: AcquisitionMethod = AcquisitionMethod.USER_IMPORT,
                        database_manager: Optional[DatabaseManager] = None) -> DatabaseIntegratedProcessor:
        """
        创建数据库集成处理器
        
        Args:
            base_processor: 基础处理器
            data_source_name: 数据源名称
            acquisition_method: 获取方式
            database_manager: 数据库管理器
            
        Returns:
            DatabaseIntegratedProcessor: 数据库集成处理器
        """
        return DatabaseIntegratedProcessor(
            base_processor=base_processor,
            database_manager=database_manager,
            data_source_name=data_source_name,
            acquisition_method=acquisition_method
        )
    
    @staticmethod
    def create_batch_processor(base_processor: BaseArticleProcessor,
                             batch_name: str,
                             acquisition_method: AcquisitionMethod = AcquisitionMethod.BATCH_CRAWL) -> DatabaseIntegratedProcessor:
        """
        创建批量处理器
        
        Args:
            base_processor: 基础处理器
            batch_name: 批次名称
            acquisition_method: 获取方式
            
        Returns:
            DatabaseIntegratedProcessor: 数据库集成处理器
        """
        return DatabaseIntegratedProcessor(
            base_processor=base_processor,
            data_source_name=f"batch_{batch_name}",
            acquisition_method=acquisition_method
        )
    
    @staticmethod
    def create_import_processor(base_processor: BaseArticleProcessor,
                              import_source: str = "user_import") -> DatabaseIntegratedProcessor:
        """
        创建导入处理器
        
        Args:
            base_processor: 基础处理器
            import_source: 导入源
            
        Returns:
            DatabaseIntegratedProcessor: 数据库集成处理器
        """
        return DatabaseIntegratedProcessor(
            base_processor=base_processor,
            data_source_name=import_source,
            acquisition_method=AcquisitionMethod.USER_IMPORT
        )