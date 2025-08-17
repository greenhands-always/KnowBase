"""
处理配置管理
提供预定义的配置模板和配置管理功能
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from src.infrastructure.utils import PathUtil


@dataclass
class ProcessingConfig:
    """处理配置类"""
    
    # LLM配置
    llm_provider: str = "ollama"  # ollama, openai, openai_compatible
    llm_model: str = "zephyr"
    llm_temperature: float = 0.1
    llm_max_tokens: Optional[int] = None
    
    # OpenAI兼容API配置
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None  # 自定义API端点，如 "http://localhost:8000/v1"
    openai_organization: Optional[str] = None
    
    # 输入配置
    input_type: str = "directory"  # directory, json_file, single_file
    input_path: str = ""
    file_pattern: str = "*.md"
    min_file_size: int = 100
    max_files: Optional[int] = None
    
    # 处理配置
    processor_type: str = "standard"  # standard, custom, basic, full
    custom_processors: List[str] = field(default_factory=list)
    enable_concept_extraction: bool = True
    enable_quality_scoring: bool = True
    enable_importance_scoring: bool = True
    enable_categorization: bool = True
    enable_tagging: bool = True
    
    # 输出配置
    output_path: str = ""
    output_format: str = "json"  # json, csv, jsonl, excel
    save_summary_report: bool = True
    
    # 性能配置
    batch_size: int = 10
    enable_progress: bool = True
    enable_parallel: bool = False
    max_workers: int = 4
    
    # 调试配置
    debug_mode: bool = False
    log_level: str = "INFO"
    save_intermediate_results: bool = False
    
    # 数据库配置
    enable_database: bool = False
    data_source_name: str = "default"
    acquisition_method: str = "user_import"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.base_dir = PathUtil.get_project_base_dir()
        self.config_dir = PathUtil.concat_path(self.base_dir, "config")
        Path(self.config_dir).mkdir(exist_ok=True)
    
    def save_config(self, config: ProcessingConfig, name: str) -> str:
        """保存配置到文件"""
        config_file = str(PathUtil.concat_path(self.config_dir, f"{name}.json"))
        
        config_dict = {
            "llm_provider": config.llm_provider,
            "llm_model": config.llm_model,
            "llm_temperature": config.llm_temperature,
            "llm_max_tokens": config.llm_max_tokens,
            "openai_api_key": config.openai_api_key,
            "openai_api_base": config.openai_api_base,
            "openai_organization": config.openai_organization,
            "input_type": config.input_type,
            "input_path": str(config.input_path),  # 转换为字符串
            "file_pattern": config.file_pattern,
            "min_file_size": config.min_file_size,
            "max_files": config.max_files,
            "processor_type": config.processor_type,
            "custom_processors": config.custom_processors,
            "enable_concept_extraction": config.enable_concept_extraction,
            "enable_quality_scoring": config.enable_quality_scoring,
            "enable_importance_scoring": config.enable_importance_scoring,
            "enable_categorization": config.enable_categorization,
            "enable_tagging": config.enable_tagging,
            "output_path": str(config.output_path),  # 转换为字符串
            "output_format": config.output_format,
            "save_summary_report": config.save_summary_report,
            "batch_size": config.batch_size,
            "enable_progress": config.enable_progress,
            "enable_parallel": config.enable_parallel,
            "max_workers": config.max_workers,
            "debug_mode": config.debug_mode,
            "log_level": config.log_level,
            "save_intermediate_results": config.save_intermediate_results,
            "enable_database": config.enable_database,
            "data_source_name": config.data_source_name,
            "acquisition_method": config.acquisition_method
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        return config_file
    
    def load_config(self, name: str) -> ProcessingConfig:
        """从文件加载配置"""
        config_file = str(PathUtil.concat_path(self.config_dir, f"{name}.json"))
        
        if not Path(config_file).exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return ProcessingConfig(**config_dict)
    
    def list_configs(self) -> List[str]:
        """列出所有可用的配置"""
        config_files = list(Path(self.config_dir).glob("*.json"))
        return [f.stem for f in config_files]
    
    def delete_config(self, name: str) -> bool:
        """删除配置文件"""
        config_file = str(PathUtil.concat_path(self.config_dir, f"{name}.json"))
        
        if Path(config_file).exists():
            Path(config_file).unlink()
            return True
        return False


class ConfigTemplates:
    """预定义配置模板"""
    
    @staticmethod
    def get_basic_config() -> ProcessingConfig:
        """基础配置 - 只进行概念提取和基本处理"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="ollama",
            llm_model="zephyr",
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")),
            processor_type="basic",
            enable_quality_scoring=False,
            enable_importance_scoring=False,
            enable_categorization=False,
            enable_tagging=False,
            output_path=str(PathUtil.concat_path(base_dir, "result/basic_processing_results.json")),
            batch_size=5
        )
    
    @staticmethod
    def get_standard_config() -> ProcessingConfig:
        """标准配置 - 完整的处理流程"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="ollama",
            llm_model="zephyr",
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")),
            processor_type="standard",
            output_path=str(PathUtil.concat_path(base_dir, "result/standard_processing_results.json")),
            batch_size=10,
            enable_progress=True
        )
    
    @staticmethod
    def get_full_config() -> ProcessingConfig:
        """完整配置 - 包含所有处理器"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="ollama",
            llm_model="zephyr",
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")),
            processor_type="full",
            output_path=str(PathUtil.concat_path(base_dir, "result/full_processing_results.json")),
            output_format="json",
            save_summary_report=True,
            batch_size=8,
            enable_progress=True,
            debug_mode=False
        )
    
    @staticmethod
    def get_analysis_config() -> ProcessingConfig:
        """分析配置 - 专注于深度分析"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="ollama",
            llm_model="zephyr",
            llm_temperature=0.0,  # 更确定性的输出
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")),
            processor_type="custom",
            custom_processors=["quality_scorer", "importance_scorer", "trending_scorer", "category_classifier"],
            output_path=str(PathUtil.concat_path(base_dir, "result/analysis_results.json")),
            output_format="json",
            save_summary_report=True,
            batch_size=5,
            enable_progress=True,
            save_intermediate_results=True
        )
    
    @staticmethod
    def get_batch_config() -> ProcessingConfig:
        """批量处理配置 - 适合大量文件处理"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="ollama",
            llm_model="zephyr",
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface")),
            file_pattern="**/*.md",  # 递归搜索所有markdown文件
            processor_type="standard",
            output_path=str(PathUtil.concat_path(base_dir, "result/batch_processing_results.json")),
            batch_size=20,
            enable_progress=True,
            enable_parallel=True,
            max_workers=4
        )
    
    @staticmethod
    def get_debug_config() -> ProcessingConfig:
        """调试配置 - 用于开发和调试"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="ollama",
            llm_model="zephyr",
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")),
            max_files=3,  # 限制文件数量
            processor_type="standard",
            output_path=str(PathUtil.concat_path(base_dir, "result/debug_results.json")),
            batch_size=1,
            enable_progress=True,
            debug_mode=True,
            log_level="DEBUG",
            save_intermediate_results=True
        )
    
    @staticmethod
    def get_openai_config() -> ProcessingConfig:
        """OpenAI配置 - 使用OpenAI API"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            llm_temperature=0.1,
            llm_max_tokens=2000,
            openai_api_key="your-openai-api-key-here",  # 需要用户设置
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")),
            processor_type="standard",
            output_path=str(PathUtil.concat_path(base_dir, "result/openai_processing_results.json")),
            batch_size=5,
            enable_progress=True
        )
    
    @staticmethod
    def get_openai_compatible_config() -> ProcessingConfig:
        """OpenAI兼容API配置 - 使用自定义API端点"""
        base_dir = PathUtil.get_project_base_dir()
        
        return ProcessingConfig(
            llm_provider="openai_compatible",
            llm_model="qwen3-32b",  # 示例模型名
            llm_temperature=0.1,
            llm_max_tokens=2000,
            openai_api_key="your-api-key-here",  # 某些兼容API可能需要
            openai_api_base="http://localhost:11434/v1",  # Ollama的OpenAI兼容端点
            input_type="directory",
            input_path=str(PathUtil.concat_path(base_dir, "result/Huggingface/Blog/Trending")),
            processor_type="standard",
            output_path=str(PathUtil.concat_path(base_dir, "result/compatible_processing_results.json")),
            batch_size=5,
            enable_progress=True
        )


def create_default_configs():
    """创建默认配置文件"""
    config_manager = ConfigManager()
    
    # 保存所有模板配置
    templates = {
        "basic": ConfigTemplates.get_basic_config(),
        "standard": ConfigTemplates.get_standard_config(),
        "full": ConfigTemplates.get_full_config(),
        "analysis": ConfigTemplates.get_analysis_config(),
        "batch": ConfigTemplates.get_batch_config(),
        "debug": ConfigTemplates.get_debug_config(),
        "openai": ConfigTemplates.get_openai_config(),
        "openai_compatible": ConfigTemplates.get_openai_compatible_config()
    }
    
    saved_configs = []
    for name, config in templates.items():
        config_file = config_manager.save_config(config, name)
        saved_configs.append((name, config_file))
    
    return saved_configs


if __name__ == "__main__":
    # 创建默认配置
    print("创建默认配置文件...")
    saved_configs = create_default_configs()
    
    print("已创建以下配置:")
    for name, file_path in saved_configs:
        print(f"- {name}: {file_path}")
    
    # 测试配置管理器
    config_manager = ConfigManager()
    
    print(f"\n可用配置: {config_manager.list_configs()}")
    
    # 测试加载配置
    standard_config = config_manager.load_config("standard")
    print(f"\n标准配置加载成功:")
    print(f"- LLM模型: {standard_config.llm_model}")
    print(f"- 输入路径: {standard_config.input_path}")
    print(f"- 输出路径: {standard_config.output_path}")
    print(f"- 处理器类型: {standard_config.processor_type}")