# 向量化SDK

这是一个功能完整的向量化SDK，支持将各种输入转储到向量数据库中，并提供高效的语义搜索功能。

## 功能特性

- **多种向量数据库支持**: 目前支持Qdrant，可扩展支持其他向量数据库
- **多种向量化模型**: 支持sentence-transformers、transformers和OpenAI API
- **自动文档分块**: 长文档自动分块处理，支持重叠分块
- **批量处理**: 高效的批量文档处理和向量化
- **语义搜索**: 支持文本查询和向量查询
- **过滤搜索**: 支持基于元数据的过滤搜索
- **配置管理**: 灵活的配置管理，支持环境变量和配置文件

## 安装依赖

```bash
# 安装基础依赖
pip install qdrant-client sentence-transformers

# 如果使用transformers模型
pip install transformers torch

# 如果使用OpenAI API
pip install openai
```

## 快速开始

### 1. 启动Qdrant服务

```bash
# 使用Docker启动Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 2. 基础使用

```python
import asyncio
from src.infrastructure.embedding import create_embedding_manager, DocumentInput

async def main():
    # 创建嵌入管理器
    manager = await create_embedding_manager(
        vector_db_host="localhost",
        vector_db_port=6333,
        collection_name="my_documents",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    try:
        # 添加文档
        documents = [
            DocumentInput(
                content="人工智能是计算机科学的一个分支",
                metadata={"category": "AI", "language": "zh"}
            ),
            DocumentInput(
                content="Machine learning is a subset of artificial intelligence",
                metadata={"category": "ML", "language": "en"}
            )
        ]
        
        await manager.add_documents(documents)
        
        # 搜索相似文档
        results = await manager.search_similar_documents("什么是AI？", top_k=5)
        
        for result in results:
            print(f"相似度: {result.score:.4f}")
            print(f"内容: {result.document.content}")
            print(f"元数据: {result.document.metadata}")
            print()
            
    finally:
        await manager.close()

# 运行
asyncio.run(main())
```

## 配置选项

### 向量数据库配置

```python
from src.infrastructure.embedding import VectorDatabaseConfig

config = VectorDatabaseConfig(
    host="localhost",
    port=6333,
    collection_name="my_collection",
    vector_size=768,
    distance_metric="cosine",  # cosine, euclidean, dot
    api_key=None,  # Qdrant API密钥（如果需要）
    timeout=30
)
```

### 向量化模型配置

```python
from src.infrastructure.embedding import EmbeddingConfig

# sentence-transformers配置
config = EmbeddingConfig(
    model_name="all-MiniLM-L6-v2",
    model_type="sentence_transformers",
    batch_size=32,
    device="cpu"  # 或 "cuda"
)

# OpenAI配置
config = EmbeddingConfig(
    model_name="text-embedding-3-small",
    model_type="openai",
    openai_api_key="your-api-key",
    batch_size=100
)
```

### 系统配置

```python
from src.infrastructure.embedding.config import EmbeddingSystemConfig

# 从环境变量加载
config = EmbeddingSystemConfig.from_env()

# 手动配置
config = EmbeddingSystemConfig(
    qdrant_host="localhost",
    qdrant_port=6333,
    default_embedding_model="all-mpnet-base-v2",
    chunk_size=1000,
    overlap_size=100,
    batch_size=32
)
```

## 高级用法

### 文档分块

```python
# 长文档会自动分块
long_document = DocumentInput(
    content="很长的文档内容...",
    metadata={"title": "长文档", "author": "作者"}
)

# 启用分块（默认启用）
await manager.add_document(long_document, chunk_document=True)

# 禁用分块
await manager.add_document(long_document, chunk_document=False)
```

### 过滤搜索

```python
# 基于元数据过滤
results = await manager.search_similar_documents(
    "查询文本",
    top_k=10,
    filter_conditions={
        "category": "AI",
        "language": "zh"
    }
)
```

### 向量搜索

```python
# 直接使用向量搜索
query_vector = await manager.embedding_service.encode_text("查询文本")
results = await manager.search_by_vector(query_vector, top_k=5)
```

### 文档管理

```python
# 获取文档
document = await manager.get_document("document_id")

# 更新文档
updated_doc = DocumentInput(
    id="document_id",
    content="更新后的内容",
    metadata={"updated": True}
)
await manager.update_document(updated_doc)

# 删除文档
await manager.delete_document("document_id")

# 批量删除
await manager.delete_documents(["id1", "id2", "id3"])
```

## 支持的模型

### Sentence Transformers模型

- `all-MiniLM-L6-v2` (384维) - 轻量级，速度快
- `all-mpnet-base-v2` (768维) - 平衡性能和质量
- `paraphrase-multilingual-MiniLM-L12-v2` - 多语言支持

### OpenAI模型

- `text-embedding-ada-002` (1536维)
- `text-embedding-3-small` (1536维)
- `text-embedding-3-large` (3072维)

## 环境变量

```bash
# Qdrant配置
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key

# 向量化模型配置
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_TYPE=sentence_transformers

# OpenAI配置
OPENAI_API_KEY=your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 处理配置
CHUNK_SIZE=1000
OVERLAP_SIZE=100
BATCH_SIZE=32
DEVICE=cpu
```

## 示例代码

查看 `examples.py` 文件获取完整的使用示例，包括：

- 基础文档添加和搜索
- 长文档分块处理
- 过滤搜索
- OpenAI API集成
- 配置管理

## 性能优化建议

1. **批量处理**: 尽量批量添加文档而不是逐个添加
2. **合适的分块大小**: 根据内容类型调整chunk_size
3. **GPU加速**: 如果有GPU，设置device="cuda"
4. **模型选择**: 根据精度和速度需求选择合适的模型
5. **连接池**: 在生产环境中考虑使用连接池

## 故障排除

### 常见问题

1. **连接Qdrant失败**
   - 检查Qdrant服务是否启动
   - 确认主机和端口配置正确

2. **模型加载失败**
   - 检查网络连接
   - 确认模型名称正确
   - 检查磁盘空间

3. **向量化失败**
   - 检查输入文本是否为空
   - 确认模型已正确加载
   - 检查设备配置（CPU/GPU）

4. **搜索结果为空**
   - 确认集合中有文档
   - 检查查询文本和文档语言是否匹配
   - 调整top_k参数

### 日志配置

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.infrastructure.embedding")
logger.setLevel(logging.DEBUG)
```

## 扩展开发

### 添加新的向量数据库

1. 继承 `VectorDatabase` 抽象类
2. 实现所有抽象方法
3. 在 `embedding_manager.py` 中注册新的数据库类型

### 添加新的向量化服务

1. 继承 `EmbeddingService` 抽象类
2. 实现 `load_model` 和 `encode_texts` 方法
3. 在 `create_embedding_service` 函数中添加新类型

## 许可证

本项目采用MIT许可证。