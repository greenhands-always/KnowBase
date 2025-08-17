# 文章处理工具

这是一个工程化的文章处理系统，基于原始的 `langchain_ner.py` 案例开发，提供可复用的工具类和组件来处理文章、提取概念、分析内容。

## 🚀 功能特性

- **概念提取**: 使用LLM从文章中提取概念、实体和关键词
- **文章处理**: 支持质量评分、重要性评分、分类、标签生成等
- **处理管道**: 流式处理大量文章，支持批量操作
- **结果管理**: 多种格式输出（JSON、CSV、Excel等）
- **配置管理**: 预定义配置模板，支持自定义配置
- **可扩展性**: 支持自定义处理器和处理流程

## 📁 项目结构

```
src/processing/
├── __init__.py              # 模块初始化
├── concept_extractor.py     # 概念提取器
├── article_processor.py     # 文章处理器
├── processing_pipeline.py   # 处理管道
├── result_manager.py        # 结果管理器
├── processors.py           # 预定义处理器
├── config.py               # 配置管理
├── main.py                 # 命令行入口
├── examples.py             # 使用示例
└── README.md               # 本文档
```

## 🛠️ 安装和设置

1. 确保已安装必要的依赖：
```bash
pip install langchain pydantic pandas openpyxl
```

2. 确保Ollama服务正在运行，并已下载zephyr模型：
```bash
ollama pull zephyr
```

## 📖 使用方法

### 1. 命令行使用

#### 初始化配置
```bash
python src/processing/main.py init-configs
```

#### 查看可用配置
```bash
python src/processing/main.py list-configs
```

#### 使用预定义配置处理
```bash
# 使用标准配置
python src/processing/main.py config standard

# 使用自定义输入输出路径
python src/processing/main.py config standard --input "d:/path/to/articles" --output "d:/path/to/results.json"
```

#### 快速处理
```bash
# 基础处理
python src/processing/main.py quick "d:/path/to/articles" --type basic

# 完整处理
python src/processing/main.py quick "d:/path/to/articles" --type full --output "results.json"
```

### 2. 编程接口使用

#### 基础使用
```python
from src.processing import ConceptExtractor, ArticleProcessor, PipelineBuilder
from src.infrastructure.utils import LLMUtil

# 创建LLM提供者
provider = LLMUtil.OllamaProvider(model_name="zephyr")
llm = provider.get_llm(model_name="zephyr")

# 创建概念提取器
concept_extractor = ConceptExtractor.create_llm_extractor(llm)

# 创建文章处理器
article_processor = ArticleProcessor.create_standard_processor(concept_extractor)

# 处理单篇文章
article_data = {
    "id": "article_1",
    "title": "AI技术发展趋势",
    "file_path": "path/to/article.md"
}

result = article_processor.process_article(article_data)
print(f"处理结果: {result.title}, 质量分: {result.quality_score}")
```

#### 使用处理管道
```python
from src.processing import PipelineBuilder

# 使用构建器创建管道
pipeline = (PipelineBuilder()
           .with_processor(article_processor)
           .with_input_directory("d:/articles", "*.md")
           .with_output("results.json", "json")
           .with_limits(max_files=10)
           .build())

# 运行管道
results = pipeline.run()
pipeline.print_summary()
```

#### 使用配置管理
```python
from src.processing import ConfigManager, ConfigTemplates

# 创建配置管理器
config_manager = ConfigManager()

# 使用模板配置
config = ConfigTemplates.get_standard_config()
config.input_path = "d:/my/articles"
config.output_path = "d:/my/results.json"

# 保存配置
config_manager.save_config(config, "my_config")

# 加载配置
loaded_config = config_manager.load_config("my_config")
```

#### 自定义处理器
```python
from src.processing.processors import BASIC_PROCESSORS

def custom_ai_scorer(result):
    """自定义AI相关性评分器"""
    if result.concepts:
        ai_keywords = ['ai', 'machine learning', 'deep learning']
        all_text = ' '.join(result.concepts.concepts + result.concepts.keywords).lower()
        
        ai_score = sum(1 for keyword in ai_keywords if keyword in all_text) / len(ai_keywords)
        result.metadata['ai_relevance_score'] = ai_score
    
    return result

# 创建使用自定义处理器的处理器
custom_processors = BASIC_PROCESSORS + [custom_ai_scorer]
article_processor = ArticleProcessor.create_custom_processor(
    concept_extractor=concept_extractor,
    processors=custom_processors
)
```

## ⚙️ 配置选项

### 预定义配置

- **basic**: 基础配置，只进行概念提取
- **standard**: 标准配置，包含完整处理流程
- **full**: 完整配置，包含所有处理器
- **analysis**: 分析配置，专注于深度分析
- **batch**: 批量配置，适合大量文件处理
- **debug**: 调试配置，用于开发调试

### 配置参数

```python
ProcessingConfig(
    # LLM配置
    llm_provider="ollama",
    llm_model="zephyr",
    llm_temperature=0.1,
    
    # 输入配置
    input_type="directory",
    input_path="path/to/articles",
    file_pattern="*.md",
    max_files=None,
    
    # 处理配置
    processor_type="standard",
    enable_concept_extraction=True,
    enable_quality_scoring=True,
    
    # 输出配置
    output_path="results.json",
    output_format="json",
    save_summary_report=True,
    
    # 性能配置
    batch_size=10,
    enable_progress=True
)
```

## 📊 输出格式

### 处理结果结构
```json
{
  "article_id": "article_1",
  "title": "文章标题",
  "status": "completed",
  "concepts": {
    "concepts": ["概念1", "概念2"],
    "entities": ["实体1", "实体2"],
    "keywords": ["关键词1", "关键词2"],
    "confidence": 0.9
  },
  "quality_score": 0.85,
  "importance_score": 0.78,
  "categories": ["AI/ML", "Technology"],
  "tags": ["AI", "技术", "趋势"],
  "metadata": {
    "processing_time": 2.5,
    "file_size": 1024
  }
}
```

### 摘要报告
```json
{
  "total_articles": 10,
  "successful_processing": 9,
  "failed_processing": 1,
  "average_quality_score": 0.82,
  "average_importance_score": 0.75,
  "top_categories": ["AI/ML", "Technology"],
  "top_concepts": ["人工智能", "机器学习"],
  "processing_time": 25.3
}
```

## 🔧 扩展开发

### 创建自定义概念提取器
```python
from src.processing.concept_extractor import BaseConceptExtractor

class CustomConceptExtractor(BaseConceptExtractor):
    def extract_from_text(self, text: str) -> ConceptExtractionResult:
        # 实现自定义提取逻辑
        pass
```

### 创建自定义处理器
```python
def custom_processor(result: ProcessingResult) -> ProcessingResult:
    # 实现自定义处理逻辑
    result.metadata['custom_field'] = "custom_value"
    return result
```

### 创建自定义管道
```python
from src.processing.processing_pipeline import ProcessingPipeline

class CustomPipeline(ProcessingPipeline):
    def process_batch(self, articles):
        # 实现自定义批处理逻辑
        pass
```

## 📝 示例

查看 `examples.py` 文件获取更多详细的使用示例：

```bash
python src/processing/examples.py
```

## 🐛 故障排除

### 常见问题

1. **LLM连接失败**
   - 确保Ollama服务正在运行
   - 检查模型是否已下载：`ollama list`

2. **文件路径错误**
   - 使用绝对路径
   - 确保文件存在且可读

3. **内存不足**
   - 减少batch_size
   - 限制max_files数量

4. **处理速度慢**
   - 启用并行处理
   - 使用更快的LLM模型

## 📄 许可证

本项目遵循MIT许可证。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！