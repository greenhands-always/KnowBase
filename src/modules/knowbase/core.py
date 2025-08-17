"""
KnowBase 核心实现
多源信息聚合中枢的核心逻辑
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, AsyncIterator
from datetime import datetime, timedelta
import aiohttp
import feedparser
from email.parser import BytesParser
from email.policy import default
import imaplib
import ssl

from src.core.models import (
    Article, RawContent, DataSource, DataSourceType, 
    ProcessingStatus, ContentType
)
from src.core.interfaces import DataCollectorInterface, DataSourceManagerInterface
from src.core.config import get_config


logger = logging.getLogger(__name__)


class RSSCollector(DataCollectorInterface):
    """RSS订阅收集器"""
    
    async def collect(self, source: DataSource) -> AsyncIterator[RawContent]:
        """收集RSS内容"""
        try:
            url = source.config.get('url')
            if not url:
                raise ValueError("RSS源URL不能为空")
            
            # 使用feedparser解析RSS
            feed = await self._parse_rss_async(url)
            
            for entry in feed.entries:
                raw_content = RawContent(
                    source_type=DataSourceType.RSS,
                    source_config=source.config,
                    raw_data={
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'description': entry.get('description', ''),
                        'published': entry.get('published', ''),
                        'author': entry.get('author', ''),
                        'tags': [tag.term for tag in entry.get('tags', [])]
                    },
                    collected_at=datetime.now()
                )
                yield raw_content
                
        except Exception as e:
            logger.error(f"收集RSS内容失败: {e}")
            raise
    
    async def _parse_rss_async(self, url: str) -> Any:
        """异步解析RSS"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, feedparser.parse, url)
    
    async def validate_source(self, source: DataSource) -> bool:
        """验证RSS源"""
        try:
            url = source.config.get('url')
            if not url:
                return False
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """获取RSS源信息"""
        return {
            'type': 'RSS',
            'description': 'RSS订阅源',
            'required_config': ['url'],
            'optional_config': ['update_interval', 'max_items']
        }


class EmailCollector(DatacollectorInterface):
    """邮件收集器"""
    
    async def collect(self, source: DataSource) -> AsyncIterator[RawContent]:
        """收集邮件内容"""
        try:
            config = source.config
            imap_server = config.get('imap_server')
            username = config.get('username')
            password = config.get('password')
            folder = config.get('folder', 'INBOX')
            
            if not all([imap_server, username, password]):
                raise ValueError("邮件配置不完整")
            
            # 连接IMAP服务器
            mail = await self._connect_imap_async(imap_server, username, password)
            
            try:
                # 选择文件夹
                mail.select(folder)
                
                # 搜索邮件
                status, messages = mail.search(None, 'ALL')
                message_ids = messages[0].split()
                
                limit = config.get('max_emails', 50)
                for msg_id in message_ids[-limit:]:
                    status, msg_data = mail.fetch(msg_id, '(RFC822)')
                    
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            email_message = BytesParser(policy=default).parsebytes(response_part[1])
                            
                            raw_content = RawContent(
                                source_type=DataSourceType.EMAIL,
                                source_config=config,
                                raw_data={
                                    'subject': email_message['subject'],
                                    'from': email_message['from'],
                                    'to': email_message['to'],
                                    'date': email_message['date'],
                                    'body': self._extract_email_body(email_message),
                                    'attachments': len(email_message.get_payload()) if email_message.is_multipart() else 0
                                },
                                collected_at=datetime.now()
                            )
                            yield raw_content
                            
            finally:
                mail.close()
                mail.logout()
                
        except Exception as e:
            logger.error(f"收集邮件内容失败: {e}")
            raise
    
    async def _connect_imap_async(self, server: str, username: str, password: str) -> Any:
        """异步连接IMAP"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._connect_imap, server, username, password)
    
    def _connect_imap(self, server: str, username: str, password: str) -> Any:
        """连接IMAP"""
        mail = imaplib.IMAP4_SSL(server)
        mail.login(username, password)
        return mail
    
    def _extract_email_body(self, email_message) -> str:
        """提取邮件正文"""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == 'text/plain':
                    return part.get_payload(decode=True).decode()
        else:
            return email_message.get_payload(decode=True).decode()
    
    async def validate_source(self, source: DataSource) -> bool:
        """验证邮件配置"""
        try:
            config = source.config
            imap_server = config.get('imap_server')
            username = config.get('username')
            password = config.get('password')
            
            if not all([imap_server, username, password]):
                return False
            
            # 测试连接
            mail = await self._connect_imap_async(imap_server, username, password)
            mail.logout()
            return True
        except Exception:
            return False
    
    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """获取邮件源信息"""
        return {
            'type': 'Email',
            'description': '邮件订阅源',
            'required_config': ['imap_server', 'username', 'password'],
            'optional_config': ['folder', 'max_emails', 'filters']
        }


class WebCollector(DatacollectorInterface):
    """网络爬虫收集器"""
    
    async def collect(self, source: DataSource) -> AsyncIterator[RawContent]:
        """收集网页内容"""
        try:
            config = source.config
            url = config.get('url')
            selector = config.get('selector', 'article')
            
            if not url:
                raise ValueError("爬虫URL不能为空")
            
            # 爬取网页内容
            content = await self._fetch_content_async(url, selector)
            
            raw_content = RawContent(
                source_type=DataSourceType.CRAWLER,
                source_config=config,
                raw_data={
                    'url': url,
                    'title': content.get('title', ''),
                    'content': content.get('content', ''),
                    'published': content.get('published', ''),
                    'author': content.get('author', ''),
                    'tags': content.get('tags', [])
                },
                collected_at=datetime.now()
            )
            yield raw_content
            
        except Exception as e:
            logger.error(f"收集网页内容失败: {e}")
            raise
    
    async def _fetch_content_async(self, url: str, selector: str) -> Dict[str, Any]:
        """异步爬取网页内容"""
        # 这里可以集成现有的爬虫代码
        # 目前使用简单的实现
        return {
            'title': f'爬取内容 - {url}',
            'content': f'从 {url} 爬取的内容',
            'published': datetime.now().isoformat(),
            'author': 'unknown',
            'tags': ['web-crawl']
        }
    
    async def validate_source(self, source: DataSource) -> bool:
        """验证爬虫配置"""
        try:
            url = source.config.get('url')
            return bool(url and url.startswith(('http://', 'https://')))
        except Exception:
            return False
    
    async def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """获取爬虫源信息"""
        return {
            'type': 'Web Crawler',
            'description': '网页爬虫源',
            'required_config': ['url'],
            'optional_config': ['selector', 'headers', 'cookies']
        }


class KnowBaseCore:
    """KnowBase核心实现"""
    
    def __init__(self):
        self.config = get_config().knowbase
        self.collectors = {
            DataSourceType.RSS: RSSCollector(),
            DataSourceType.EMAIL: EmailCollector(),
            DataSourceType.CRAWLER: WebCollector(),
        }
        self._active_sources = {}
    
    def register_collector(self, source_type: DataSourceType, collector: DataCollectorInterface):
        """注册收集器"""
        self.collectors[source_type] = collector
    
    async def collect_from_source(self, source: DataSource) -> List[RawContent]:
        """从指定数据源收集内容"""
        if source.type not in self.collectors:
            raise ValueError(f"不支持的数据源类型: {source.type}")
        
        collector = self.collectors[source.type]
        contents = []
        
        async for content in collector.collect(source):
            contents.append(content)
        
        return contents
    
    async def validate_source(self, source: DataSource) -> bool:
        """验证数据源"""
        if source.type not in self.collectors:
            return False
        
        collector = self.collectors[source.type]
        return await collector.validate_source(source)
    
    async def sync_all_sources(self, sources: List[DataSource]) -> Dict[str, Any]:
        """同步所有数据源"""
        results = {
            'total_sources': len(sources),
            'successful': 0,
            'failed': 0,
            'errors': {}
        }
        
        for source in sources:
            try:
                if source.is_active:
                    contents = await self.collect_from_source(source)
                    results['successful'] += 1
                    logger.info(f"数据源 {source.name} 同步成功，收集 {len(contents)} 条内容")
                else:
                    logger.info(f"数据源 {source.name} 已禁用，跳过")
                    
            except Exception as e:
                results['failed'] += 1
                results['errors'][source.id] = str(e)
                logger.error(f"数据源 {source.name} 同步失败: {e}")
        
        return results
    
    async def get_source_info(self, source_type: DataSourceType) -> Dict[str, Any]:
        """获取数据源类型信息"""
        if source_type not in self.collectors:
            return {'error': '不支持的数据源类型'}
        
        collector = self.collectors[source_type]
        return await collector.get_source_info('')
    
    async def should_sync_source(self, source: DataSource) -> bool:
        """判断是否应该同步数据源"""
        if not source.is_active:
            return False
        
        if not source.last_sync:
            return True
        
        if not source.sync_interval:
            return False
        
        next_sync = source.last_sync + timedelta(minutes=source.sync_interval)
        return datetime.now() >= next_sync
    
    def get_supported_source_types(self) -> List[str]:
        """获取支持的数据源类型"""
        return [source_type.value for source_type in self.collectors.keys()]