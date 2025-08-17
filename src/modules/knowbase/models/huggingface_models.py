"""
Hugging Face相关数据模型定义
"""

from typing import List, Dict, Optional
from datetime import datetime


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