from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from ...infrastructure.utils.db.DatabaseManager import DatabaseManager, AcquisitionMethod
from ..processing.article_processor import ProcessingResult
from ..models.models import Article, DataSource

class ImporterType(str, Enum):
    """导入器类型枚举"""
    FOLDER = "folder"
    URL = "url"
    RSS = "rss"
    EMAIL = "email"
    API = "api"
    DATABASE = "database"
    CLIPBOARD = "clipboard"

@dataclass
class ImportResult:
    """导入结果"""
    success: bool
    total_items: int
    imported_items: int
    failed_items: int
    duplicate_items: int
    batch_id: str
    errors: List[str]
    article_ids: List[int]  # 成功导入的文章ID列表
    processing_time: float
    
class BaseImporter(ABC):
    """导入器抽象基类"""
    
    def __init__(self, 
                 database_manager: Optional[DatabaseManager] = None,
                 data_source_name: str = "default",
                 acquisition_method: AcquisitionMethod = AcquisitionMethod.USER_IMPORT,
                 auto_process: bool = True,
                 deduplication_enabled: bool = True):
        """
        初始化导入器
        
        Args:
            database_manager: 数据库管理器
            data_source_name: 数据源名称
            acquisition_method: 获取方式
            auto_process: 是否自动处理
            deduplication_enabled: 是否启用去重
        """
        self.db_manager = database_manager or DatabaseManager()
        self.data_source_name = data_source_name
        self.acquisition_method = acquisition_method
        self.auto_process = auto_process
        self.deduplication_enabled = deduplication_enabled
        
        # 确保数据源存在
        self._ensure_data_source()
    
    def _ensure_data_source(self):
        """确保数据源存在"""
        try:
            self.data_source_id = self.db_manager.create_data_source(
                name=self.data_source_name,
                source_type=self.get_importer_type().value,
                acquisition_method=self.acquisition_method,
                description=f"Data source for {self.get_importer_type().value} importer"
            )
        except Exception as e:
            # 如果数据源已存在，获取其ID
            self.data_source_id = self.db_manager.get_data_source_id(self.data_source_name)
    
    @abstractmethod
    def get_importer_type(self) -> ImporterType:
        """获取导入器类型"""
        pass
    
    @abstractmethod
    def extract_content(self, source: Any) -> Generator[Dict[str, Any], None, None]:
        """从源提取内容"""
        pass
    
    def import_data(self, source: Any, user_id: Optional[int] = None) -> ImportResult:
        """导入数据的主要方法"""
        start_time = datetime.now()
        batch_id = self._generate_batch_id()
        
        # 创建导入批次记录
        self._create_import_batch(batch_id, user_id)
        
        total_items = 0
        imported_items = 0
        failed_items = 0
        duplicate_items = 0
        errors = []
        article_ids = []
        
        try:
            # 提取内容
            for content_data in self.extract_content(source):
                total_items += 1
                
                try:
                    # 插入到数据库
                    result = self._insert_article_to_database(content_data, batch_id, user_id)
                    
                    if result['success']:
                        if result['is_duplicate']:
                            duplicate_items += 1
                        else:
                            imported_items += 1
                            article_ids.append(result['article_id'])
                            
                            # 如果启用自动处理，创建处理任务
                            if self.auto_process:
                                self._create_processing_tasks(result['article_id'])
                    else:
                        failed_items += 1
                        errors.append(result['error'])
                        
                except Exception as e:
                    failed_items += 1
                    errors.append(f"处理项目失败: {str(e)}")
            
            # 更新批次状态
            self._update_import_batch(batch_id, total_items, imported_items, 
                                    failed_items, duplicate_items, 'completed')
            
        except Exception as e:
            errors.append(f"导入过程失败: {str(e)}")
            self._update_import_batch(batch_id, total_items, imported_items, 
                                    failed_items, duplicate_items, 'failed')
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ImportResult(
            success=failed_items == 0,
            total_items=total_items,
            imported_items=imported_items,
            failed_items=failed_items,
            duplicate_items=duplicate_items,
            batch_id=batch_id,
            errors=errors,
            article_ids=article_ids,
            processing_time=processing_time
        )
    
    def _insert_article_to_database(self, content_data: Dict[str, Any], 
                                   batch_id: str, user_id: Optional[int]) -> Dict[str, Any]:
        """将文章插入数据库"""
        try:
            # 创建内容指纹
            fingerprint_id = self.db_manager.create_content_fingerprint(
                content=content_data.get('content', ''),
                title=content_data.get('title', ''),
                url=content_data.get('url', ''),
                metadata={
                    'word_count': len(content_data.get('content', '').split()),
                    'language': content_data.get('language', 'en')
                }
            )
            
            # 检查重复
            if self.deduplication_enabled:
                duplicate_article_id = self.db_manager.check_duplicate_by_fingerprint(fingerprint_id)
                if duplicate_article_id:
                    return {
                        'success': True,
                        'is_duplicate': True,
                        'article_id': duplicate_article_id,
                        'error': None
                    }
            
            # 插入文章
            article_id = self.db_manager.insert_article(
                source_id=self.data_source_id,
                title=content_data.get('title', ''),
                url=content_data.get('url'),
                author=content_data.get('author'),
                published_at=content_data.get('published_at'),
                acquisition_method=self.acquisition_method,
                fingerprint_id=fingerprint_id,
                summary=content_data.get('summary'),
                language=content_data.get('language', 'en'),
                word_count=len(content_data.get('content', '').split()),
                imported_by=user_id,
                import_batch_id=batch_id,
                acquisition_metadata=content_data.get('metadata', {})
            )
            
            # 存储原始内容到MongoDB（如果需要）
            if content_data.get('content'):
                mongo_id = self._store_raw_content_to_mongo(article_id, content_data)
                self.db_manager.update_article_mongo_id(article_id, mongo_id)
            
            return {
                'success': True,
                'is_duplicate': False,
                'article_id': article_id,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'is_duplicate': False,
                'article_id': None,
                'error': str(e)
            }
    
    def _create_processing_tasks(self, article_id: int):
        """为文章创建处理任务"""
        # 这里可以调用处理队列系统
        # 数据库触发器会自动创建基础任务，这里可以添加额外的任务
        pass
    
    def _generate_batch_id(self) -> str:
        """生成批次ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.get_importer_type().value}_{timestamp}"
    
    def _create_import_batch(self, batch_id: str, user_id: Optional[int]):
        """创建导入批次记录"""
        self.db_manager.create_import_batch(
            batch_id=batch_id,
            imported_by=user_id,
            import_source=self.get_importer_type().value,
            source_description=f"Import from {self.get_importer_type().value}",
            processing_config={
                'auto_process': self.auto_process,
                'deduplication_enabled': self.deduplication_enabled
            },
            deduplication_enabled=self.deduplication_enabled,
            auto_process=self.auto_process
        )
    
    def _update_import_batch(self, batch_id: str, total: int, successful: int, 
                           failed: int, duplicate: int, status: str):
        """更新导入批次状态"""
        self.db_manager.update_import_batch(
            batch_id=batch_id,
            total_items=total,
            successful_items=successful,
            failed_items=failed,
            duplicate_items=duplicate,
            status=status
        )
    
    def _store_raw_content_to_mongo(self, article_id: int, content_data: Dict[str, Any]) -> str:
        """存储原始内容到MongoDB（需要实现MongoDB连接）"""
        # 这里需要实现MongoDB存储逻辑
        # 返回MongoDB文档ID
        return f"mongo_{article_id}_{datetime.now().timestamp()}"