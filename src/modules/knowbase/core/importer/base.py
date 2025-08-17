"""
导入器抽象基类
定义所有导入器的通用接口和行为
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, Callable
from pathlib import Path
from datetime import datetime
from enum import Enum

from ...models.models import Article, RawContent, DataSourceType, ProcessingStatus


class ImporterType(str, Enum):
    """导入器类型枚举"""
    FOLDER = "folder"
    FILE = "file"
    URL = "url"
    RSS = "rss"
    EMAIL = "email"
    API = "api"
    DATABASE = "database"


class ImportResult:
    """导入结果类"""
    def __init__(self):
        self.success_count: int = 0
        self.failed_count: int = 0
        self.skipped_count: int = 0
        self.articles: List[Article] = []
        self.errors: List[Dict[str, Any]] = []
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        
    def add_success(self, article: Article):
        """添加成功导入的文章"""
        self.success_count += 1
        self.articles.append(article)
        
    def add_error(self, source: str, error: str):
        """添加错误记录"""
        self.failed_count += 1
        self.errors.append({
            "source": source,
            "error": error,
            "timestamp": datetime.now()
        })
        
    def add_skipped(self, source: str, reason: str):
        """添加跳过记录"""
        self.skipped_count += 1
        self.errors.append({
            "source": source,
            "reason": reason,
            "type": "skipped",
            "timestamp": datetime.now()
        })
        
    def finish(self):
        """标记导入完成"""
        self.end_time = datetime.now()
        
    @property
    def total_processed(self) -> int:
        return self.success_count + self.failed_count + self.skipped_count
        
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class BaseImporter(ABC):
    """导入器抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.importer_type = self._get_importer_type()
        self.progress_callback: Optional[Callable[[int, int], None]] = None
        
    @abstractmethod
    def _get_importer_type(self) -> ImporterType:
        """获取导入器类型"""
        pass
        
    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        pass
        
    @abstractmethod
    def discover_sources(self) -> List[str]:
        """发现可导入的源"""
        pass
        
    @abstractmethod
    def extract_content(self, source: str) -> Optional[RawContent]:
        """从源提取原始内容"""
        pass
        
    @abstractmethod
    def parse_content(self, raw_content: RawContent) -> Optional[Article]:
        """解析原始内容为文章"""
        pass
        
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
        
    def _notify_progress(self, current: int, total: int):
        """通知进度更新"""
        if self.progress_callback:
            self.progress_callback(current, total)
            
    def import_all(self) -> ImportResult:
        """执行完整导入流程"""
        result = ImportResult()
        
        try:
            # 验证配置
            if not self.validate_config():
                result.add_error("config", "配置验证失败")
                result.finish()
                return result
                
            # 发现源
            sources = self.discover_sources()
            if not sources:
                result.add_error("discovery", "未发现可导入的源")
                result.finish()
                return result
                
            # 处理每个源
            for i, source in enumerate(sources):
                try:
                    # 提取原始内容
                    raw_content = self.extract_content(source)
                    if not raw_content:
                        result.add_skipped(source, "无法提取内容")
                        continue
                        
                    # 解析为文章
                    article = self.parse_content(raw_content)
                    if not article:
                        result.add_skipped(source, "无法解析内容")
                        continue
                        
                    # 添加成功记录
                    result.add_success(article)
                    
                except Exception as e:
                    result.add_error(source, str(e))
                    
                finally:
                    self._notify_progress(i + 1, len(sources))
                    
        except Exception as e:
            result.add_error("import_all", f"导入过程发生错误: {str(e)}")
            
        result.finish()
        return result
        
    def import_single(self, source: str) -> Optional[Article]:
        """导入单个源"""
        try:
            raw_content = self.extract_content(source)
            if not raw_content:
                return None
                
            return self.parse_content(raw_content)
            
        except Exception:
            return None