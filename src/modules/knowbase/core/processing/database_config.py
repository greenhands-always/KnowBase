"""
数据库配置模板
为不同场景提供预配置的数据库处理配置
"""

from typing import Dict, Any
from datetime import datetime

from .config import ProcessingConfig
from .database_processor import AcquisitionMethod


class DatabaseConfigTemplates:
    """数据库配置模板"""
    
    @staticmethod
    def get_file_import_config() -> ProcessingConfig:
        """文件导入配置"""
        config = ProcessingConfig(
            # 基础配置
            llm_provider="openai_compatible",
            llm_model="qwen-turbo-latest",
            processor_type="standard",
            
            # 输入输出
            input_type="directory",
            input_path="result/Huggingface/Blog",
            file_pattern="*.md",
            output_path=f"results_db_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            output_format="json",
            
            # 数据库配置
            enable_database=True,
            data_source_name="file_import",
            acquisition_method=AcquisitionMethod.USER_IMPORT,
            
            # 处理配置
            enable_concept_extraction=True,
            enable_summary_generation=True,
            enable_classification=True,
            enable_quality_assessment=True,
            
            # 其他配置
            enable_progress=True,
            save_summary_report=True,
            max_files=50
        )
        return config
    
    @staticmethod
    def get_batch_crawl_config() -> ProcessingConfig:
        """批量爬取配置"""
        config = ProcessingConfig(
            # 基础配置
            llm_provider="openai_compatible",
            llm_model="qwen-turbo-latest",
            processor_type="full",
            
            # 输入输出
            input_type="directory",
            input_path="data/crawled",
            file_pattern="*.json",
            output_path=f"results_batch_crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            output_format="json",
            
            # 数据库配置
            enable_database=True,
            data_source_name="batch_crawl",
            acquisition_method=AcquisitionMethod.BATCH_CRAWL,
            
            # 处理配置
            enable_concept_extraction=True,
            enable_summary_generation=True,
            enable_classification=True,
            enable_quality_assessment=True,
            enable_deduplication=True,
            
            # 其他配置
            enable_progress=True,
            save_summary_report=True,
            max_files=200
        )
        return config
    
    @staticmethod
    def get_stream_processing_config() -> ProcessingConfig:
        """流式处理配置"""
        config = ProcessingConfig(
            # 基础配置
            llm_provider="openai_compatible",
            llm_model="qwen-turbo-latest",
            processor_type="fast",
            
            # 输入输出
            input_type="stream",
            output_path=f"results_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            output_format="jsonl",
            
            # 数据库配置
            enable_database=True,
            data_source_name="stream_processing",
            acquisition_method=AcquisitionMethod.STREAM_CRAWL,
            
            # 处理配置
            enable_concept_extraction=True,
            enable_summary_generation=False,  # 流式处理跳过摘要生成
            enable_classification=True,
            enable_quality_assessment=True,
            enable_deduplication=True,
            
            # 其他配置
            enable_progress=False,  # 流式处理不显示进度
            save_summary_report=False,
            max_files=1000
        )
        return config
    
    @staticmethod
    def get_api_sync_config() -> ProcessingConfig:
        """API同步配置"""
        config = ProcessingConfig(
            # 基础配置
            llm_provider="openai_compatible",
            llm_model="qwen-turbo-latest",
            processor_type="standard",
            
            # 输入输出
            input_type="api",
            output_path=f"results_api_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            output_format="json",
            
            # 数据库配置
            enable_database=True,
            data_source_name="api_sync",
            acquisition_method=AcquisitionMethod.API_SYNC,
            
            # 处理配置
            enable_concept_extraction=True,
            enable_summary_generation=True,
            enable_classification=True,
            enable_quality_assessment=True,
            enable_deduplication=True,
            
            # 其他配置
            enable_progress=True,
            save_summary_report=True,
            max_files=100
        )
        return config
    
    @staticmethod
    def get_manual_entry_config() -> ProcessingConfig:
        """手动录入配置"""
        config = ProcessingConfig(
            # 基础配置
            llm_provider="openai_compatible",
            llm_model="qwen-turbo-latest",
            processor_type="full",
            
            # 输入输出
            input_type="manual",
            output_path=f"results_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            output_format="json",
            
            # 数据库配置
            enable_database=True,
            data_source_name="manual_entry",
            acquisition_method=AcquisitionMethod.MANUAL_ENTRY,
            
            # 处理配置
            enable_concept_extraction=True,
            enable_summary_generation=True,
            enable_classification=True,
            enable_quality_assessment=True,
            enable_deduplication=False,  # 手动录入通常不需要去重
            
            # 其他配置
            enable_progress=True,
            save_summary_report=True,
            max_files=10
        )
        return config
    
    @staticmethod
    def create_custom_database_config(
        data_source_name: str,
        acquisition_method: AcquisitionMethod,
        input_path: str,
        processor_type: str = "standard",
        **kwargs
    ) -> ProcessingConfig:
        """
        创建自定义数据库配置
        
        Args:
            data_source_name: 数据源名称
            acquisition_method: 获取方式
            input_path: 输入路径
            processor_type: 处理器类型
            **kwargs: 其他配置参数
            
        Returns:
            ProcessingConfig: 处理配置
        """
        config = ProcessingConfig(
            # 基础配置
            llm_provider=kwargs.get("llm_provider", "openai_compatible"),
            llm_model=kwargs.get("llm_model", "qwen-turbo-latest"),
            processor_type=processor_type,
            
            # 输入输出
            input_type=kwargs.get("input_type", "directory"),
            input_path=input_path,
            file_pattern=kwargs.get("file_pattern", "*.md"),
            output_path=kwargs.get("output_path", f"results_{data_source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
            output_format=kwargs.get("output_format", "json"),
            
            # 数据库配置
            enable_database=True,
            data_source_name=data_source_name,
            acquisition_method=acquisition_method,
            
            # 处理配置
            enable_concept_extraction=kwargs.get("enable_concept_extraction", True),
            enable_summary_generation=kwargs.get("enable_summary_generation", True),
            enable_classification=kwargs.get("enable_classification", True),
            enable_quality_assessment=kwargs.get("enable_quality_assessment", True),
            enable_deduplication=kwargs.get("enable_deduplication", True),
            
            # 其他配置
            enable_progress=kwargs.get("enable_progress", True),
            save_summary_report=kwargs.get("save_summary_report", True),
            max_files=kwargs.get("max_files", 100)
        )
        return config


def create_database_configs():
    """创建数据库配置文件"""
    import json
    from pathlib import Path
    
    # 配置目录
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # 创建各种数据库配置
    configs = {
        "db_file_import": DatabaseConfigTemplates.get_file_import_config(),
        "db_batch_crawl": DatabaseConfigTemplates.get_batch_crawl_config(),
        "db_stream_processing": DatabaseConfigTemplates.get_stream_processing_config(),
        "db_api_sync": DatabaseConfigTemplates.get_api_sync_config(),
        "db_manual_entry": DatabaseConfigTemplates.get_manual_entry_config()
    }
    
    saved_configs = []
    
    for name, config in configs.items():
        file_path = config_dir / f"{name}.json"
        
        # 转换为字典并处理枚举类型
        config_dict = config.model_dump()
        if 'acquisition_method' in config_dict:
            config_dict['acquisition_method'] = config_dict['acquisition_method'].value
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        
        saved_configs.append((name, str(file_path)))
    
    return saved_configs


if __name__ == "__main__":
    # 创建数据库配置文件
    saved = create_database_configs()
    print("已创建数据库配置文件:")
    for name, path in saved:
        print(f"  {name}: {path}")