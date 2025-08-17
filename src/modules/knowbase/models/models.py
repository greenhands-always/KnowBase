"""
统一数据模型定义
为所有PKM Copilot模块提供标准化的数据接口
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class ContentType(str, Enum):
    """内容类型枚举"""
    ARTICLE = "article"
    EMAIL = "email"
    RSS = "rss"
    SOCIAL = "social"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"


class ProcessingStatus(str, Enum):
    """处理状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DataSourceType(str, Enum):
    """数据源类型枚举"""
    RSS = "rss"
    EMAIL = "email"
    CRAWLER = "crawler"
    API = "api"
    MANUAL = "manual"
    IMPORT = "import"


class EntityType(str, Enum):
    """实体类型枚举"""
    CONCEPT = "concept"
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"


class Article(BaseModel):
    """统一文章模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    content: str
    summary: Optional[str] = None
    source: str
    url: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    collected_at: datetime = Field(default_factory=datetime.now)
    
    # 内容分类
    content_type: ContentType = ContentType.ARTICLE
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    
    # 语义信息
    concepts: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    
    # 向量化信息
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # 质量评分
    quality_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # 处理状态
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_error: Optional[str] = None
    
    # 关联信息
    related_articles: List[str] = Field(default_factory=list)
    knowledge_graph_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RawContent(BaseModel):
    """原始内容模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: DataSourceType
    source_config: Dict[str, Any]
    raw_data: Dict[str, Any]
    collected_at: datetime = Field(default_factory=datetime.now)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING


class ProcessingResult(BaseModel):
    """处理结果模型"""
    article_id: str
    status: ProcessingStatus
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    concepts_extracted: List[str] = Field(default_factory=list)
    keywords_extracted: List[str] = Field(default_factory=list)
    summary_generated: Optional[str] = None
    embedding_generated: bool = False
    processed_at: datetime = Field(default_factory=datetime.now)


class Entity(BaseModel):
    """知识图谱实体模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: EntityType
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Relationship(BaseModel):
    """知识图谱关系模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    target_id: str
    type: str
    strength: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class VectorData(BaseModel):
    """向量数据模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    article_id: str
    content: str
    embedding: List[float]
    embedding_model: str
    content_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class FilterConfig(BaseModel):
    """过滤配置模型"""
    min_quality_score: Optional[float] = None
    min_relevance_score: Optional[float] = None
    required_tags: List[str] = Field(default_factory=list)
    excluded_tags: List[str] = Field(default_factory=list)
    date_range: Optional[Dict[str, datetime]] = None
    source_whitelist: List[str] = Field(default_factory=list)
    source_blacklist: List[str] = Field(default_factory=list)


class SummaryConfig(BaseModel):
    """总结配置模型"""
    summary_type: str = "brief"  # brief, detailed, bullet_points
    max_length: int = 500
    include_keywords: bool = True
    include_concepts: bool = True
    output_format: str = "markdown"  # markdown, json, html
    language: str = "zh"


class Collection(BaseModel):
    """收藏模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    article_ids: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_public: bool = False
    owner_id: Optional[str] = None


class SearchQuery(BaseModel):
    """搜索查询模型"""
    query: str
    filters: Optional[FilterConfig] = None
    limit: int = 10
    offset: int = 0
    search_type: str = "semantic"  # semantic, keyword, hybrid
    sort_by: str = "relevance"
    include_vectors: bool = False


class SearchResult(BaseModel):
    """搜索结果模型"""
    article: Article
    score: float
    highlights: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field(default_factory=list)


class ProcessingJob(BaseModel):
    """处理任务模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    source_config: Dict[str, Any]
    processing_config: Dict[str, Any]
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    results_count: int = 0


class DataSource(BaseModel):
    """数据源模型"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: DataSourceType
    config: Dict[str, Any]
    is_active: bool = True
    last_sync: Optional[datetime] = None
    sync_interval: Optional[int] = None  # 分钟
    created_at: datetime = Field(default_factory=datetime.now)


class SyncStatus(BaseModel):
    """同步状态模型"""
    source_id: str
    last_sync: datetime
    next_sync: Optional[datetime] = None
    articles_synced: int = 0
    errors: List[str] = Field(default_factory=list)
    is_successful: bool = True
    

class HuggingFaceArticle:
    """
    表示 Hugging Face 平台上的文章或博客内容

    属性:
        title (str): 文章标题
        author (str): 作者用户名
        published_date (datetime.date): 发布日期
        last_updated (datetime.date): 最后更新日期
        url (str): 文章完整URL
        tags (List[str]): 文章标签列表
        categories (List[str]): 文章分类列表
        summary (str): 文章摘要
        content (str): 文章完整内容(HTML或Markdown)
        read_time (int): 预计阅读时间(分钟)
        views (int): 阅读次数
        likes (int): 点赞数
        comments (List[Dict]): 评论列表
        featured (bool): 是否精选文章
        cover_image (str): 封面图片URL
        references (List[str]): 引用资源列表
        huggingface_resources (List[Dict]): 关联的Hugging Face资源(模型/数据集/空间)
    """

    def __init__(self,
                 title: str,
                 author: str,
                 published_date: datetime.date,
                 url: str,
                 content: str = "",
                 summary: str = "",
                 tags: Optional[List[str]] = None,
                 categories: Optional[List[str]] = None,
                 last_updated: Optional[datetime.date] = None,
                 read_time: int = 0,
                 views: int = 0,
                 likes: int = 0,
                 comments: Optional[List[Dict]] = None,
                 featured: bool = False,
                 cover_image: str = "",
                 references: Optional[List[str]] = None,
                 huggingface_resources: Optional[List[Dict]] = None):
        """
        初始化 HuggingFaceArticle 实例

        参数:
            title: 文章标题
            author: 作者用户名
            published_date: 发布日期
            url: 文章完整URL
            content: 文章完整内容
            summary: 文章摘要
            tags: 文章标签列表
            categories: 文章分类列表
            last_updated: 最后更新日期
            read_time: 预计阅读时间(分钟)
            views: 阅读次数
            likes: 点赞数
            comments: 评论列表
            featured: 是否精选文章
            cover_image: 封面图片URL
            references: 引用资源列表
            huggingface_resources: 关联的Hugging Face资源
        """
        self.title = title
        self.author = author
        self.published_date = published_date
        self.url = url
        self.content = content
        self.summary = summary
        self.tags = tags if tags else []
        self.categories = categories if categories else []
        self.last_updated = last_updated if last_updated else published_date
        self.read_time = read_time
        self.views = views
        self.likes = likes
        self.comments = comments if comments else []
        self.featured = featured
        self.cover_image = cover_image
        self.references = references if references else []
        self.huggingface_resources = huggingface_resources if huggingface_resources else []

    def add_comment(self, username: str, text: str, date: datetime.date):
        """
        添加评论到文章

        参数:
            username: 评论者用户名
            text: 评论内容
            date: 评论日期
        """
        self.comments.append({
            "username": username,
            "text": text,
            "date": date
        })

    def add_resource(self, resource_type: str, name: str, url: str):
        """
        添加关联的Hugging Face资源

        参数:
            resource_type: 资源类型(model/dataset/space)
            name: 资源名称
            url: 资源URL
        """
        self.huggingface_resources.append({
            "type": resource_type,
            "name": name,
            "url": url
        })

    def to_dict(self) -> Dict:
        """
        将文章对象转换为字典

        返回:
            包含所有属性的字典
        """
        return {
            "title": self.title,
            "author": self.author,
            "published_date": self.published_date.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "url": self.url,
            "tags": self.tags,
            "categories": self.categories,
            "summary": self.summary,
            "content": self.content,
            "read_time": self.read_time,
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "featured": self.featured,
            "cover_image": self.cover_image,
            "references": self.references,
            "huggingface_resources": self.huggingface_resources
        }

    def __repr__(self) -> str:
        """
        返回对象的字符串表示
        """
        return (f"<HuggingFaceArticle(title='{self.title}', "
                f"author='{self.author}', "
                f"published_date={self.published_date}, "
                f"url='{self.url}')>")

    @classmethod
    def from_dict(cls, data: Dict) -> 'HuggingFaceArticle':
        """
        从字典创建HuggingFaceArticle实例

        参数:
            data: 包含文章数据的字典

        返回:
            HuggingFaceArticle实例
        """
        # 转换日期字符串为日期对象
        published_date = datetime.datetime.strptime(data['published_date'], '%Y-%m-%d').date()
        last_updated = datetime.datetime.strptime(data['last_updated'], '%Y-%m-%d').date()

        return cls(
            title=data['title'],
            author=data['author'],
            published_date=published_date,
            last_updated=last_updated,
            url=data['url'],
            tags=data.get('tags', []),
            categories=data.get('categories', []),
            summary=data.get('summary', ''),
            content=data.get('content', ''),
            read_time=data.get('read_time', 0),
            views=data.get('views', 0),
            likes=data.get('likes', 0),
            comments=data.get('comments', []),
            featured=data.get('featured', False),
            cover_image=data.get('cover_image', ''),
            references=data.get('references', []),
            huggingface_resources=data.get('huggingface_resources', [])
        )

    def display_summary(self) -> str:
        """
        返回文章的摘要信息字符串
        """
        return (f"标题: {self.title}\n"
                f"作者: {self.author}\n"
                f"发布日期: {self.published_date}\n"
                f"标签: {', '.join(self.tags)}\n"
                f"阅读时间: {self.read_time} 分钟\n"
                f"阅读量: {self.views} 次\n"
                f"点赞数: {self.likes}\n"
                f"摘要: {self.summary[:200]}...")