# PKM Copilot 模块化架构

## 架构概览

基于完整产品构想，我们将系统重构为6个核心模块，每个模块可以独立使用，也可以协同工作：

1. **KnowBase (知库)** - 多源信息聚合中枢
2. **VecEmbed (向量工坊)** - 多模态信息向量化引擎
3. **SiftFlow (筛流)** - AI智能过滤系统
4. **SumAgent (总结代理)** - 自动化内容提炼工具
5. **LinkVerse (关联宇宙)** - 可视化知识图谱模块
6. **CollectDeck (收藏甲板)** - 个人知识交互入口

## 模块间协作流程

```
KnowBase → VecEmbed → SiftFlow → SumAgent → LinkVerse → CollectDeck
   ↓        ↓        ↓        ↓        ↓        ↓
原始数据   语义向量   过滤提纯   结构化内容  知识关联   用户交互
```

## 技术架构

### 数据流架构
- **事件驱动**: 使用消息队列处理异步任务
- **分层存储**: Raw Data (MongoDB) → Processed Data (PostgreSQL) → Vector (Pinecone/Weaviate) → Cache (Redis)
- **统一接口**: 所有模块通过标准API进行通信

### 技术栈选择
- **Python + LangChain**: 负责AI流程核心
- **FastAPI**: 提供RESTful API接口
- **PostgreSQL**: 结构化数据存储
- **MongoDB**: 非结构化数据存储
- **Qdrant/Pinecone**: 向量数据库存储
- **Neo4j**: 知识图谱存储

## 模块详细设计

### 1. KnowBase (知库)
- **功能**: 多源信息聚合，支持RSS、邮件订阅、社交媒体、爬虫等
- **输入**: 各种信息源
- **输出**: 统一格式的原始数据
- **存储**: MongoDB (原始数据) + PostgreSQL (元数据)

### 2. VecEmbed (向量工坊)
- **功能**: 将文本、音频等内容转化为语义向量
- **输入**: 原始内容
- **输出**: 向量表示
- **存储**: Qdrant/Pinecone
- **接口**: MCP标准接口

### 3. SiftFlow (筛流)
- **功能**: 基于AI的内容质量过滤
- **输入**: 原始内容 + 用户偏好
- **输出**: 过滤后的高质量内容
- **算法**: 内容质量评分 + 用户偏好匹配

### 4. SumAgent (总结代理)
- **功能**: 自动化内容提炼和摘要生成
- **输入**: 过滤后的内容
- **输出**: 结构化摘要、主题标签、报告
- **配置**: 支持用户自定义摘要格式

### 5. LinkVerse (关联宇宙)
- **功能**: 知识图谱构建和可视化
- **输入**: 结构化内容
- **输出**: 实体关系图
- **存储**: Neo4j
- **接口**: GraphQL API

### 6. CollectDeck (收藏甲板)
- **功能**: 用户交互入口，支持收藏、批注、搜索
- **输入**: 用户操作
- **输出**: 个性化知识库
- **功能**: 语义搜索、批注、导出

## 核心数据模型

### 统一文章模型
```python
class Article(BaseModel):
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    tags: List[str]
    concepts: List[str]
    vectors: Optional[List[float]]
    metadata: Dict[str, Any]
```

### 知识图谱模型
```python
class Entity(BaseModel):
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any]

class Relationship(BaseModel):
    id: str
    source_id: str
    target_id: str
    type: str
    strength: float
    metadata: Dict[str, Any]
```

## 接口设计

### RESTful API标准
- `/api/v1/knowbase/sources` - 数据源管理
- `/api/v1/vecembed/vectors` - 向量化服务
- `/api/v1/siftflow/filter` - 内容过滤
- `/api/v1/sumagent/summarize` - 内容总结
- `/api/v1/linkverse/graph` - 知识图谱
- `/api/v1/collectdeck/collections` - 收藏管理

### MCP接口标准
每个模块都提供MCP标准接口，支持第三方集成。

## 配置管理

### 模块配置
```yaml
modules:
  knowbase:
    enabled: true
    sources:
      - rss
      - email
      - crawler
  vecembed:
    enabled: true
    provider: qdrant
    model: all-MiniLM-L6-v2
  siftflow:
    enabled: true
    quality_threshold: 0.8
  sumagent:
    enabled: true
    summary_type: structured
  linkverse:
    enabled: true
    graph_store: neo4j
  collectdeck:
    enabled: true
    search_backend: semantic
```

## 部署架构

### 本地部署
- 所有模块打包为Python包
- 支持pip安装和Docker部署
- 提供CLI工具

### 云端部署
- 微服务架构
- Kubernetes容器编排
- 自动扩缩容
- 负载均衡

## 开发规范

### 代码结构
```
src/
├── modules/
│   ├── knowbase/
│   │   ├── __init__.py
│   │   ├── core.py
│   │   ├── api.py
│   │   ├── models.py
│   │   └── tests/
│   ├── vecembed/
│   ├── siftflow/
│   ├── sumagent/
│   ├── linkverse/
│   └── collectdeck/
├── core/
│   ├── models.py
│   ├── interfaces.py
│   └── config.py
└── tests/
    ├── integration/
    └── unit/
```

### 测试策略
- 单元测试：每个模块独立测试
- 集成测试：模块间协作测试
- 端到端测试：完整工作流测试

## 迁移计划

### 阶段1：基础重构
1. 创建模块基础结构
2. 迁移现有功能到新模块
3. 建立统一接口

### 阶段2：功能增强
1. 实现缺失的模块功能
2. 完善API接口
3. 添加配置管理

### 阶段3：优化完善
1. 性能优化
2. 用户体验改进
3. 文档完善