from typing import Dict, Type, Any, Optional

from .base_importer import BaseImporter, ImporterType
from .folder_importer import FolderImporter
from ...infrastructure.utils.db.DatabaseManager import DatabaseManager, AcquisitionMethod

class ImporterFactory:
    """导入器工厂"""
    
    _importers: Dict[ImporterType, Type[BaseImporter]] = {
        ImporterType.FOLDER: FolderImporter,
        # 可以在这里注册其他导入器
    }
    
    @classmethod
    def create_importer(cls, 
                       importer_type: ImporterType,
                       database_manager: Optional[DatabaseManager] = None,
                       **kwargs) -> BaseImporter:
        """创建导入器实例"""
        if importer_type not in cls._importers:
            raise ValueError(f"不支持的导入器类型: {importer_type}")
        
        importer_class = cls._importers[importer_type]
        return importer_class(database_manager=database_manager, **kwargs)
    
    @classmethod
    def register_importer(cls, importer_type: ImporterType, 
                         importer_class: Type[BaseImporter]):
        """注册新的导入器类型"""
        cls._importers[importer_type] = importer_class
    
    @classmethod
    def get_available_importers(cls) -> List[ImporterType]:
        """获取可用的导入器类型"""
        return list(cls._importers.keys())