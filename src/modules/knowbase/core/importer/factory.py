"""
导入器工厂
根据配置创建相应的导入器实例
"""

from typing import Dict, Any, Optional
from .base import BaseImporter, ImporterType
from .folder_importer import FolderImporter


class ImporterFactory:
    """导入器工厂类"""
    
    _importers = {
        ImporterType.FOLDER: FolderImporter,
        # 未来可以添加更多导入器
        # ImporterType.URL: UrlImporter,
        # ImporterType.RSS: RssImporter,
        # ImporterType.EMAIL: EmailImporter,
    }
    
    @classmethod
    def create_importer(cls, importer_type: ImporterType, config: Dict[str, Any]) -> Optional[BaseImporter]:
        """创建导入器实例"""
        importer_class = cls._importers.get(importer_type)
        if not importer_class:
            return None
            
        return importer_class(config)
        
    @classmethod
    def get_supported_types(cls) -> list[ImporterType]:
        """获取支持的导入器类型"""
        return list(cls._importers.keys())
        
    @classmethod
    def register_importer(cls, importer_type: ImporterType, importer_class):
        """注册新的导入器类型"""
        cls._importers[importer_type] = importer_class