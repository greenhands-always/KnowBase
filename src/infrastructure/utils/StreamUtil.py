import os
import json
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class StreamItem:
    """流处理项的数据结构"""
    id: str
    title: str
    file_path: Path
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FileStreamBuilder:
    """文件流构建器，用于从文件系统构建数据流"""
    
    def __init__(self, base_dir: Path):
        """
        初始化文件流构建器
        
        Args:
            base_dir: 基础目录路径
        """
        self.base_dir = Path(base_dir)
        self.filters = []
        self.transformers = []
        self.limit = None  # 添加limit属性
        
    def add_filter(self, filter_func: Callable[[Path], bool]) -> 'FileStreamBuilder':
        """
        添加文件过滤器
        
        Args:
            filter_func: 过滤函数，接收文件路径，返回是否包含该文件
        """
        self.filters.append(filter_func)
        return self
        
    def add_transformer(self, transform_func: Callable[[StreamItem], StreamItem]) -> 'FileStreamBuilder':
        """
        添加数据转换器
        
        Args:
            transform_func: 转换函数，接收StreamItem，返回转换后的StreamItem
        """
        self.transformers.append(transform_func)
        return self
        
    def set_limit(self, limit: int) -> 'FileStreamBuilder':
        """
        设置返回项目的最大数量限制
        
        Args:
            limit: 最大返回数量，None表示无限制
        """
        self.limit = limit if limit is None or limit > 0 else None
        return self
        
    def build_from_directory(self, pattern: str = "*.md") -> Iterator[StreamItem]:
        """
        从目录构建文件流
        
        Args:
            pattern: 文件匹配模式，默认为 "*.md"
            
        Yields:
            StreamItem: 流处理项
        """
        if not self.base_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.base_dir}")
            
        count = 0  # 添加计数器
        
        # 遍历目录中的文件
        for file_path in self.base_dir.glob(pattern):
            # 检查是否达到限制
            if self.limit is not None and count >= self.limit:
                break
                
            if not file_path.is_file():
                continue
                
            # 应用过滤器
            if not all(filter_func(file_path) for filter_func in self.filters):
                continue
                
            # 创建流项
            item = StreamItem(
                id=file_path.stem,
                title=file_path.stem,
                file_path=file_path,
                metadata={
                    'file_size': file_path.stat().st_size,
                    'modified_time': file_path.stat().st_mtime
                }
            )
            
            # 应用转换器
            for transformer in self.transformers:
                item = transformer(item)
                
            yield item
            count += 1  # 增加计数器
            
    def build_from_json_with_files(self, json_file: Path, file_dir: Path) -> Iterator[StreamItem]:
        """
        从JSON文件和对应的文件目录构建流
        
        Args:
            json_file: JSON文件路径
            file_dir: 文件目录路径
            
        Yields:
            StreamItem: 流处理项
        """
        if not json_file.exists():
            raise FileNotFoundError(f"JSON文件不存在: {json_file}")
            
        if not file_dir.exists():
            raise FileNotFoundError(f"文件目录不存在: {file_dir}")
            
        # 读取JSON数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0  # 添加计数器
        
        # 遍历JSON中的每个项目
        for post in data:
            # 检查是否达到限制
            if self.limit is not None and count >= self.limit:
                break
                
            post_title = post.get("title", "")
            if not post_title:
                continue
                
            # 查找对应的文件
            file_path = file_dir / f"{post_title}.md"
            
            # 应用过滤器
            if not all(filter_func(file_path) for filter_func in self.filters):
                continue
                
            if file_path.exists():
                # 创建流项
                item = StreamItem(
                    id=post_title,
                    title=post_title,
                    file_path=file_path,
                    metadata={
                        'json_data': post,
                        'file_size': file_path.stat().st_size if file_path.exists() else 0,
                        'modified_time': file_path.stat().st_mtime if file_path.exists() else 0
                    }
                )
                
                # 应用转换器
                for transformer in self.transformers:
                    item = transformer(item)
                    
                yield item
                count += 1  # 增加计数器


class StreamProcessor:
    """流处理器，用于处理数据流"""
    
    def __init__(self):
        self.processors = []
        
    def add_processor(self, processor_func: Callable[[StreamItem], Any]) -> 'StreamProcessor':
        """
        添加处理器函数
        
        Args:
            processor_func: 处理函数，接收StreamItem，返回处理结果
        """
        self.processors.append(processor_func)
        return self
        
    def process_stream(self, stream: Iterator[StreamItem], 
                      progress_callback: Optional[Callable[[int, str], None]] = None) -> List[Any]:
        """
        处理数据流
        
        Args:
            stream: 数据流
            progress_callback: 进度回调函数，接收(当前索引, 当前项目标题)
            
        Returns:
            List[Any]: 处理结果列表
        """
        results = []
        
        for idx, item in enumerate(stream):
            if progress_callback:
                progress_callback(idx, item.title)
                
            # 应用所有处理器
            item_results = {}
            for processor in self.processors:
                try:
                    result = processor(item)
                    item_results[processor.__name__] = result
                except Exception as e:
                    print(f"处理项目 {item.title} 时出错: {e}")
                    item_results[processor.__name__] = None
                    
            results.append({
                'item': item,
                'results': item_results
            })
            
        return results


# 常用的过滤器函数
def file_exists_filter(file_path: Path) -> bool:
    """文件存在过滤器"""
    return file_path.exists()


def file_size_filter(min_size: int = 0, max_size: int = float('inf')) -> Callable[[Path], bool]:
    """文件大小过滤器"""
    def filter_func(file_path: Path) -> bool:
        if not file_path.exists():
            return False
        size = file_path.stat().st_size
        return min_size <= size <= max_size
    return filter_func


def file_extension_filter(extensions: List[str]) -> Callable[[Path], bool]:
    """文件扩展名过滤器"""
    def filter_func(file_path: Path) -> bool:
        return file_path.suffix.lower() in [ext.lower() for ext in extensions]
    return filter_func


# 常用的转换器函数
def add_content_transformer(item: StreamItem) -> StreamItem:
    """添加文件内容的转换器"""
    if item.file_path.exists():
        try:
            with open(item.file_path, 'r', encoding='utf-8') as f:
                item.metadata['content'] = f.read()
        except Exception as e:
            print(f"读取文件 {item.file_path} 内容时出错: {e}")
            item.metadata['content'] = ""
    return item


if __name__ == "__main__":
    # 示例用法
    from pathlib import Path
    
    # 创建文件流构建器
    builder = FileStreamBuilder(Path("./test_dir"))
    
    # 添加过滤器和转换器
    stream = (builder
              .add_filter(file_exists_filter)
              .add_filter(file_size_filter(min_size=100))
              .add_transformer(add_content_transformer)
              .build_from_directory("*.md"))
    
    # 创建流处理器
    processor = StreamProcessor()
    
    def example_processor(item: StreamItem) -> str:
        return f"处理了文件: {item.title}"
    
    processor.add_processor(example_processor)
    
    # 处理流
    results = processor.process_stream(stream, 
                                     progress_callback=lambda idx, title: print(f"正在处理第{idx+1}个文件: {title}"))
    
    for result in results:
        print(result)