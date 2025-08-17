"""
数据库管理器
负责将文章处理过程和结果存储到PostgreSQL数据库
"""

import hashlib
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import logging

from .PostgresConnector import PostgresConnector
from ....processing.article_processor import ProcessingResult, ProcessingStatus


class AcquisitionMethod(str, Enum):
    """数据获取方式枚举"""
    BATCH_CRAWL = "batch_crawl"
    STREAM_CRAWL = "stream_crawl"
    USER_IMPORT = "user_import"
    API_SYNC = "api_sync"
    MANUAL_ENTRY = "manual_entry"


class ProcessingStage(str, Enum):
    """处理阶段枚举"""
    INGESTED = "ingested"
    DEDUPLICATED = "deduplicated"
    PREPROCESSED = "preprocessed"
    CONTENT_EXTRACTED = "content_extracted"
    CONCEPTS_EXTRACTED = "concepts_extracted"
    CLASSIFIED = "classified"
    RELATIONS_EXTRACTED = "relations_extracted"
    SUMMARIZED = "summarized"
    QUALITY_ASSESSED = "quality_assessed"
    INDEXED = "indexed"
    COMPLETED = "completed"


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, connector: Optional[PostgresConnector] = None):
        """
        初始化数据库管理器
        
        Args:
            connector: PostgreSQL连接器实例
        """
        self.connector = connector or PostgresConnector()
        self.logger = logging.getLogger(__name__)
    
    def _get_connection(self):
        """获取数据库连接"""
        return self.connector.connect()
    
    def _generate_content_hash(self, content: str) -> str:
        """生成内容哈希"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _generate_title_hash(self, title: str) -> str:
        """生成标题哈希"""
        return hashlib.sha256(title.encode('utf-8')).hexdigest()
    
    def _generate_url_hash(self, url: str) -> str:
        """生成URL哈希"""
        # 标准化URL后生成哈希
        normalized_url = url.lower().strip().rstrip('/')
        return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()
    
    def create_data_source(self, name: str, source_type: str, 
                          acquisition_method: AcquisitionMethod,
                          base_url: Optional[str] = None,
                          description: Optional[str] = None,
                          config: Optional[Dict] = None) -> int:
        """
        创建数据源
        
        Args:
            name: 数据源名称
            source_type: 数据源类型
            acquisition_method: 获取方式
            base_url: 基础URL
            description: 描述
            config: 配置信息
            
        Returns:
            int: 数据源ID
        """
        conn = self._get_connection()
        if not conn:
            raise Exception("无法连接到数据库")
        
        try:
            with conn.cursor() as cursor:
                query = """
                INSERT INTO data_sources (name, type, acquisition_method, base_url, description, config)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    type = EXCLUDED.type,
                    acquisition_method = EXCLUDED.acquisition_method,
                    base_url = EXCLUDED.base_url,
                    description = EXCLUDED.description,
                    config = EXCLUDED.config,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING id
                """
                cursor.execute(query, (
                    name, source_type, acquisition_method.value, 
                    base_url, description, Json(config or {})
                ))
                result = cursor.fetchone()
                conn.commit()
                return result[0]
        except Exception as e:
            conn.rollback()
            self.logger.error(f"创建数据源失败: {e}")
            raise
    
    def create_content_fingerprint(self, content: str, title: str, 
                                 url: Optional[str] = None,
                                 author: Optional[str] = None,
                                 metadata: Optional[Dict] = None) -> int:
        """
        创建内容指纹
        
        Args:
            content: 文章内容
            title: 文章标题
            url: 文章URL
            author: 作者
            metadata: 元数据
            
        Returns:
            int: 指纹ID
        """
        conn = self._get_connection()
        if not conn:
            raise Exception("无法连接到数据库")
        
        try:
            with conn.cursor() as cursor:
                # 生成各种哈希
                content_hash = self._generate_content_hash(content)
                title_hash = self._generate_title_hash(title)
                url_hash = self._generate_url_hash(url) if url else None
                
                # 生成组合哈希
                author_title_text = f"{author or ''}|{title}"
                author_title_hash = hashlib.sha256(author_title_text.encode('utf-8')).hexdigest()
                
                # 计算文本统计
                word_count = len(content.split())
                char_count = len(content)
                
                query = """
                INSERT INTO content_fingerprints 
                (content_hash, title_hash, url_hash, author_title_hash, 
                 word_count, char_count, language)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (content_hash) DO UPDATE SET
                    title_hash = EXCLUDED.title_hash,
                    url_hash = EXCLUDED.url_hash,
                    author_title_hash = EXCLUDED.author_title_hash
                RETURNING id
                """
                cursor.execute(query, (
                    content_hash, title_hash, url_hash, author_title_hash,
                    word_count, char_count, metadata.get('language', 'en') if metadata else 'en'
                ))
                result = cursor.fetchone()
                conn.commit()
                return result[0]
        except Exception as e:
            conn.rollback()
            self.logger.error(f"创建内容指纹失败: {e}")
            raise
    
    def create_article(self, title: str, source_id: int,
                      acquisition_method: AcquisitionMethod,
                      url: Optional[str] = None,
                      author: Optional[str] = None,
                      published_at: Optional[datetime] = None,
                      content: Optional[str] = None,
                      external_id: Optional[str] = None,
                      fingerprint_id: Optional[int] = None,
                      metadata: Optional[Dict] = None) -> int:
        """
        创建文章记录
        
        Args:
            title: 文章标题
            source_id: 数据源ID
            acquisition_method: 获取方式
            url: 文章URL
            author: 作者
            published_at: 发布时间
            content: 文章内容
            external_id: 外部ID
            fingerprint_id: 指纹ID
            metadata: 元数据
            
        Returns:
            int: 文章ID
        """
        conn = self._get_connection()
        if not conn:
            raise Exception("无法连接到数据库")
        
        try:
            with conn.cursor() as cursor:
                # 如果提供了内容但没有指纹ID，创建指纹
                if content and not fingerprint_id:
                    fingerprint_id = self.create_content_fingerprint(
                        content, title, url, author, metadata
                    )
                
                # 计算基础统计
                word_count = len(content.split()) if content else None
                reading_time = word_count // 200 if word_count else None  # 假设每分钟200词
                
                query = """
                INSERT INTO articles 
                (source_id, external_id, title, url, author, published_at,
                 acquisition_method, fingerprint_id, word_count, reading_time_minutes,
                 acquisition_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """
                cursor.execute(query, (
                    source_id, external_id, title, url, author, published_at,
                    acquisition_method.value, fingerprint_id, word_count, reading_time,
                    Json(metadata or {})
                ))
                result = cursor.fetchone()
                article_id = result[0]
                conn.commit()
                
                # 创建初始处理任务
                self._create_initial_processing_tasks(article_id)
                
                return article_id
        except Exception as e:
            conn.rollback()
            self.logger.error(f"创建文章失败: {e}")
            raise
    
    def _create_initial_processing_tasks(self, article_id: int):
        """为新文章创建初始处理任务"""
        conn = self._get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cursor:
                # 创建处理状态记录
                stages = [
                    ProcessingStage.CONTENT_EXTRACTED,
                    ProcessingStage.CONCEPTS_EXTRACTED,
                    ProcessingStage.SUMMARIZED,
                    ProcessingStage.CLASSIFIED,
                    ProcessingStage.QUALITY_ASSESSED
                ]
                
                for stage in stages:
                    query = """
                    INSERT INTO processing_status (article_id, stage, status)
                    VALUES (%s, %s, 'pending')
                    ON CONFLICT (article_id, stage) DO NOTHING
                    """
                    cursor.execute(query, (article_id, stage.value))
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"创建初始处理任务失败: {e}")
    
    def update_processing_status(self, article_id: int, stage: ProcessingStage,
                               status: str, result_data: Optional[Dict] = None,
                               error_message: Optional[str] = None,
                               processing_time: Optional[float] = None,
                               confidence_score: Optional[float] = None,
                               processor_name: Optional[str] = None) -> bool:
        """
        更新处理状态
        
        Args:
            article_id: 文章ID
            stage: 处理阶段
            status: 状态
            result_data: 结果数据
            error_message: 错误信息
            processing_time: 处理时间
            confidence_score: 置信度
            processor_name: 处理器名称
            
        Returns:
            bool: 是否成功
        """
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cursor:
                # 更新处理状态
                query = """
                UPDATE processing_status SET
                    status = %s,
                    result_data = %s,
                    error_message = %s,
                    processing_time_seconds = %s,
                    confidence_score = %s,
                    processor_name = %s,
                    completed_at = CASE WHEN %s IN ('completed', 'failed') THEN CURRENT_TIMESTAMP ELSE completed_at END,
                    started_at = CASE WHEN started_at IS NULL THEN CURRENT_TIMESTAMP ELSE started_at END
                WHERE article_id = %s AND stage = %s
                """
                cursor.execute(query, (
                    status, Json(result_data) if result_data else None,
                    error_message, processing_time, confidence_score, processor_name,
                    status, article_id, stage.value
                ))
                
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            conn.rollback()
            self.logger.error(f"更新处理状态失败: {e}")
            return False
    
    def save_processing_result(self, processing_result: ProcessingResult,
                             article_id: Optional[int] = None) -> bool:
        """
        保存处理结果到数据库
        
        Args:
            processing_result: 处理结果
            article_id: 文章ID（如果不提供，会尝试从结果中获取）
            
        Returns:
            bool: 是否成功
        """
        conn = self._get_connection()
        if not conn:
            return False
        
        try:
            with conn.cursor() as cursor:
                # 如果没有提供article_id，尝试查找
                if not article_id:
                    query = "SELECT id FROM articles WHERE title = %s LIMIT 1"
                    cursor.execute(query, (processing_result.title,))
                    result = cursor.fetchone()
                    if result:
                        article_id = result[0]
                    else:
                        self.logger.error(f"找不到文章: {processing_result.title}")
                        return False
                
                # 更新文章的评分信息
                update_query = """
                UPDATE articles SET
                    summary = %s,
                    quality_score = %s,
                    importance_score = %s,
                    trending_score = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """
                cursor.execute(update_query, (
                    processing_result.summary,
                    processing_result.quality_score,
                    processing_result.importance_score,
                    processing_result.trending_score,
                    article_id
                ))
                
                # 保存概念提取结果
                if processing_result.concepts:
                    concept_data = {
                        'concepts': processing_result.concepts.concepts,
                        'entities': processing_result.concepts.entities,
                        'keywords': processing_result.concepts.keywords,
                        'confidence': processing_result.concepts.confidence
                    }
                    
                    self.update_processing_status(
                        article_id, ProcessingStage.CONCEPTS_EXTRACTED,
                        'completed', concept_data,
                        processing_time=processing_result.processing_time,
                        confidence_score=processing_result.concepts.confidence
                    )
                
                # 保存分类结果
                if processing_result.categories:
                    category_data = {
                        'categories': processing_result.categories,
                        'tags': processing_result.tags
                    }
                    
                    self.update_processing_status(
                        article_id, ProcessingStage.CLASSIFIED,
                        'completed', category_data,
                        processing_time=processing_result.processing_time
                    )
                
                # 保存质量评估结果
                if processing_result.quality_score is not None:
                    quality_data = {
                        'quality_score': processing_result.quality_score,
                        'importance_score': processing_result.importance_score,
                        'trending_score': processing_result.trending_score
                    }
                    
                    self.update_processing_status(
                        article_id, ProcessingStage.QUALITY_ASSESSED,
                        'completed', quality_data,
                        processing_time=processing_result.processing_time
                    )
                
                # 如果处理完成，更新总体状态
                if processing_result.status == ProcessingStatus.COMPLETED:
                    self.update_processing_status(
                        article_id, ProcessingStage.COMPLETED,
                        'completed', {'metadata': processing_result.metadata},
                        processing_time=processing_result.processing_time
                    )
                
                conn.commit()
                return True
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"保存处理结果失败: {e}")
            return False
    
    def get_article_processing_status(self, article_id: int) -> Dict[str, Any]:
        """
        获取文章处理状态
        
        Args:
            article_id: 文章ID
            
        Returns:
            Dict: 处理状态信息
        """
        conn = self._get_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                query = """
                SELECT 
                    a.id, a.title, a.url, a.author,
                    a.quality_score, a.importance_score, a.trending_score,
                    ps.stage, ps.status, ps.result_data, ps.error_message,
                    ps.processing_time_seconds, ps.confidence_score,
                    ps.started_at, ps.completed_at
                FROM articles a
                LEFT JOIN processing_status ps ON a.id = ps.article_id
                WHERE a.id = %s
                ORDER BY ps.stage
                """
                cursor.execute(query, (article_id,))
                results = cursor.fetchall()
                
                if not results:
                    return {}
                
                # 组织返回数据
                article_info = {
                    'id': results[0]['id'],
                    'title': results[0]['title'],
                    'url': results[0]['url'],
                    'author': results[0]['author'],
                    'quality_score': results[0]['quality_score'],
                    'importance_score': results[0]['importance_score'],
                    'trending_score': results[0]['trending_score'],
                    'processing_stages': {}
                }
                
                for row in results:
                    if row['stage']:
                        article_info['processing_stages'][row['stage']] = {
                            'status': row['status'],
                            'result_data': row['result_data'],
                            'error_message': row['error_message'],
                            'processing_time_seconds': row['processing_time_seconds'],
                            'confidence_score': row['confidence_score'],
                            'started_at': row['started_at'],
                            'completed_at': row['completed_at']
                        }
                
                return article_info
                
        except Exception as e:
            self.logger.error(f"获取文章处理状态失败: {e}")
            return {}
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Returns:
            Dict: 统计信息
        """
        conn = self._get_connection()
        if not conn:
            return {}
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # 总体统计
                query = """
                SELECT 
                    COUNT(*) as total_articles,
                    COUNT(CASE WHEN quality_score IS NOT NULL THEN 1 END) as processed_articles,
                    AVG(quality_score) as avg_quality_score,
                    AVG(importance_score) as avg_importance_score,
                    AVG(trending_score) as avg_trending_score
                FROM articles
                """
                cursor.execute(query)
                overall_stats = cursor.fetchone()
                
                # 处理状态统计
                query = """
                SELECT 
                    stage,
                    status,
                    COUNT(*) as count
                FROM processing_status
                GROUP BY stage, status
                ORDER BY stage, status
                """
                cursor.execute(query)
                status_stats = cursor.fetchall()
                
                # 组织返回数据
                stats = {
                    'overall': dict(overall_stats),
                    'processing_status': {}
                }
                
                for row in status_stats:
                    stage = row['stage']
                    if stage not in stats['processing_status']:
                        stats['processing_status'][stage] = {}
                    stats['processing_status'][stage][row['status']] = row['count']
                
                return stats
                
        except Exception as e:
            self.logger.error(f"获取处理统计失败: {e}")
            return {}
    
    def close(self):
        """关闭数据库连接"""
        if self.connector:
            self.connector.close()