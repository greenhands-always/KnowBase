# 大模型概念提取器

这个模块提供了基于大模型的技术概念提取功能，相比传统的NLP方法（如pyhanlp_ner.py），能够更准确、更全面地提取技术文章中的专业术语和概念。

## 文件说明

### 1. enhanced_concept_extractor.py
增强版概念提取器，支持多种大模型提供商，具有更好的准确性和灵活性。

### 2. llm_concept_extractor.py
基础版大模型概念提取器，使用OpenAI API。

### 3. llm_client.py
统一的LLM客户端，支持多种提供商：
- OpenAI (GPT-4, GPT-3.5-turbo等)
- Anthropic (Claude-3系列)
- Google Gemini
- Ollama (本地模型)

### 4. config.py
配置管理，支持环境变量配置。

## 使用方法

### 环境配置

设置API密钥：
```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google Gemini
export GEMINI_API_KEY="your-gemini-key"

# Ollama (本地)
export OLLAMA_BASE_URL="http://localhost:11434"
```

### 快速开始

#### 1. 使用增强版提取器
```python
from enhanced_concept_extractor import EnhancedConceptExtractor

# 创建提取器
extractor = EnhancedConceptExtractor(
    provider="openai",
    model="gpt-4o-mini"
)

# 处理目录
concepts = extractor.process_directory("path/to/markdown/files", limit=10)

# 保存结果
extractor.save_results(concepts, "output_filename")
```

#### 2. 命令行使用
```bash
# 基础使用
python enhanced_concept_extractor.py --directory ./markdown_files

# 指定提供商和模型
python enhanced_concept_extractor.py --provider openai --model gpt-4o --limit 20

# 使用Anthropic Claude
python enhanced_concept_extractor.py --provider anthropic --model claude-3-haiku-20240307

# 使用Ollama本地模型
python enhanced_concept_extractor.py --provider ollama --model llama3.1 --rate-limit 0.5
```

#### 3. 不同提供商的使用示例

**OpenAI:**
```python
extractor = EnhancedConceptExtractor(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-key"
)
```

**Anthropic Claude:**
```python
extractor = EnhancedConceptExtractor(
    provider="anthropic",
    model="claude-3-haiku-20240307"
)
```

**Google Gemini:**
```python
extractor = EnhancedConceptExtractor(
    provider="gemini",
    model="gemini-pro"
)
```

**Ollama (本地):**
```python
extractor = EnhancedConceptExtractor(
    provider="ollama",
    model="llama3.1",
    base_url="http://localhost:11434"
)
```

## 输出文件

运行后会生成以下文件：
- `enhanced_concepts_[provider]_[timestamp].csv` - 主要结果，包含概念和频率
- `enhanced_concepts_[provider]_[timestamp]_details.json` - 详细分析数据
- `enhanced_concepts_[provider]_[timestamp]_summary.md` - 分析报告

## 优势对比

| 特性 | pyhanlp_ner.py | 大模型提取器 |
|------|----------------|--------------|
| **准确性** | 基于规则，有限 | 语义理解，更准确 |
| **概念类型** | 仅限英文实体 | 中英文技术术语 |
| **上下文理解** | 无 | 有 |
| **过滤能力** | 简单 | 智能过滤 |
| **扩展性** | 困难 | 容易 |
| **多语言** | 英文为主 | 支持多语言 |

## 性能优化

### 1. 批处理
使用 `--parallel` 参数启用并发处理（注意API限制）：
```bash
python enhanced_concept_extractor.py --parallel --max-workers 3 --rate-limit 2.0
```

### 2. 内容分割
对于长文章，自动分割处理避免token限制。

### 3. 结果缓存
重复处理相同文件时使用缓存，避免重复API调用。

### 4. 智能过滤
自动过滤无意义的概念，提高结果质量。

## 故障排除

### 常见问题

1. **API密钥问题**
   ```bash
   # 检查环境变量
   echo $OPENAI_API_KEY
   ```

2. **网络连接**
   ```bash
   # 测试Ollama
   curl http://localhost:11434/api/tags
   ```

3. **模型选择**
   - 免费/低成本：gpt-4o-mini, claude-3-haiku
   - 高性能：gpt-4o, claude-3-sonnet
   - 本地：llama3.1, mistral

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

extractor = EnhancedConceptExtractor(...)
```

## 示例输出

```csv
概念,频率,占比
Transformer,25,12.5%
PyTorch,18,9.0%
Hugging Face,15,7.5%
LLM,20,10.0%
Fine-tuning,12,6.0%
```

## 注意事项

1. **API费用**：使用商业API会产生费用，建议使用前了解定价
2. **速率限制**：遵守API提供商的速率限制
3. **隐私**：注意处理敏感内容时的隐私保护
4. **本地模型**：Ollama适合隐私敏感场景，无需API密钥