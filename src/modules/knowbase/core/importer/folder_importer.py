import os
from pathlib import Path
from typing import Dict, Any, Generator, List, Optional
import mimetypes
from datetime import datetime

from .base_importer import BaseImporter, ImporterType
from ..processing.content_extractors import get_content_extractor

class FolderImporter(BaseImporter):
    """文件夹导入器"""
    
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.pdf', '.docx', '.doc', 
        '.html', '.htm', '.json', '.xml', '.csv'
    }
    
    def __init__(self, 
                 recursive: bool = True,
                 file_filters: Optional[List[str]] = None,
                 **kwargs):
        """
        初始化文件夹导入器
        
        Args:
            recursive: 是否递归扫描子文件夹
            file_filters: 文件扩展名过滤器
        """
        super().__init__(**kwargs)
        self.recursive = recursive
        self.file_filters = file_filters or list(self.SUPPORTED_EXTENSIONS)
    
    def get_importer_type(self) -> ImporterType:
        return ImporterType.FOLDER
    
    def extract_content(self, folder_path: str) -> Generator[Dict[str, Any], None, None]:
        """从文件夹提取内容"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"文件夹不存在或不是有效目录: {folder_path}")
        
        # 获取文件列表
        files = self._get_files(folder_path)
        
        for file_path in files:
            try:
                content_data = self._extract_file_content(file_path)
                if content_data:
                    yield content_data
            except Exception as e:
                # 记录错误但继续处理其他文件
                print(f"处理文件 {file_path} 时出错: {e}")
                continue
    
    def _get_files(self, folder_path: Path) -> List[Path]:
        """获取文件列表"""
        files = []
        
        if self.recursive:
            pattern = '**/*'
        else:
            pattern = '*'
        
        for file_path in folder_path.glob(pattern):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.file_filters):
                files.append(file_path)
        
        return sorted(files)
    
    def _extract_file_content(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """提取单个文件的内容"""
        try:
            # 获取文件信息
            stat = file_path.stat()
            
            # 根据文件类型选择提取器
            extractor = get_content_extractor(file_path.suffix)
            if not extractor:
                return None
            
            # 提取内容
            content = extractor.extract(str(file_path))
            
            # 生成标题（如果没有提供）
            title = content.get('title') or file_path.stem
            
            return {
                'title': title,
                'content': content.get('content', ''),
                'author': content.get('author'),
                'published_at': datetime.fromtimestamp(stat.st_mtime),
                'url': f"file://{file_path.absolute()}",
                'language': self._detect_language(content.get('content', '')),
                'summary': content.get('summary'),
                'metadata': {
                    'file_path': str(file_path),
                    'file_size': stat.st_size,
                    'file_type': file_path.suffix,
                    'mime_type': mimetypes.guess_type(str(file_path))[0],
                    'created_at': datetime.fromtimestamp(stat.st_ctime),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime),
                    **content.get('metadata', {})
                }
            }
            
        except Exception as e:
            raise Exception(f"提取文件内容失败: {e}")
    
    def _detect_language(self, content: str) -> str:
        """检测内容语言"""
        # 简单的语言检测逻辑
        if not content:
            return 'en'
        
        # 检测中文字符
        chinese_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        if chinese_chars > len(content) * 0.1:
            return 'zh'
        
        return 'en'