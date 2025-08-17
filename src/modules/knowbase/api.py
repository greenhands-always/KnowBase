"""
KnowBase API接口
提供RESTful API和数据源管理功能
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from .core import KnowBaseCore
from .models import (
    DataSourceExtended, RSSConfig, EmailConfig, WebCrawlerConfig,
    SyncResult, SourceStatistics, SourceHealth, CollectionSchedule
)
from src.core.models import DataSource, DataSourceType, ProcessingStatus


class SourceCreateRequest(BaseModel):
    """创建数据源请求"""
    name: str
    type: DataSourceType
    config: Dict[str, Any]
    description: Optional[str] = None
    sync_interval: Optional[int] = None
    is_active: bool = True


class SourceUpdateRequest(BaseModel):
    """更新数据源请求"""
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    sync_interval: Optional[int] = None
    is_active: Optional[bool] = None


class SyncRequest(BaseModel):
    """同步请求"""
    source_ids: Optional[List[str]] = None  # 如果为空，同步所有
    force: bool = False


class KnowBaseAPI:
    """KnowBase API实现"""
    
    def __init__(self):
        self.core = KnowBaseCore()
        self.app = FastAPI(title="KnowBase API", description="多源信息聚合中枢")
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.get("/sources", response_model=List[DataSourceExtended])
        async def list_sources():
            """列出所有数据源"""
            # 这里应该查询数据库，现在返回空列表
            return []
        
        @self.app.post("/sources", response_model=DataSourceExtended)
        async def create_source(request: SourceCreateRequest):
            """创建数据源"""
            source = DataSource(
                name=request.name,
                type=request.type,
                config=request.config,
                description=request.description,
                sync_interval=request.sync_interval,
                is_active=request.is_active
            )
            
            # 验证数据源
            is_valid = await self.core.validate_source(source)
            if not is_valid:
                raise HTTPException(status_code=400, detail="数据源配置无效")
            
            # 保存到数据库
            # TODO: 实际的数据库保存逻辑
            extended_source = DataSourceExtended(**source.dict())
            return extended_source
        
        @self.app.get("/sources/{source_id}", response_model=DataSourceExtended)
        async def get_source(source_id: str):
            """获取数据源详情"""
            # TODO: 从数据库查询
            raise HTTPException(status_code=404, detail="数据源不存在")
        
        @self.app.put("/sources/{source_id}", response_model=DataSourceExtended)
        async def update_source(source_id: str, request: SourceUpdateRequest):
            """更新数据源"""
            # TODO: 更新数据库中的数据源
            raise HTTPException(status_code=404, detail="数据源不存在")
        
        @self.app.delete("/sources/{source_id}")
        async def delete_source(source_id: str):
            """删除数据源"""
            # TODO: 从数据库删除数据源
            return {"message": "数据源已删除"}
        
        @self.app.post("/sources/{source_id}/validate")
        async def validate_source(source_id: str):
            """验证数据源"""
            # TODO: 获取数据源并验证
            return {"valid": True, "errors": []}
        
        @self.app.post("/sources/{source_id}/sync", response_model=SyncResult)
        async def sync_source(source_id: str, background_tasks: BackgroundTasks):
            """同步单个数据源"""
            # TODO: 实现同步逻辑
            return SyncResult(
                source_id=source_id,
                source_name="示例源",
                success=True,
                items_collected=0,
                sync_time=datetime.now()
            )
        
        @self.app.post("/sync", response_model=Dict[str, Any])
        async def sync_all(request: SyncRequest):
            """同步所有数据源"""
            # TODO: 实现批量同步
            return {
                "total_sources": 0,
                "successful": 0,
                "failed": 0,
                "errors": {}
            }
        
        @self.app.get("/sources/{source_id}/statistics", response_model=SourceStatistics)
        async def get_source_statistics(source_id: str):
            """获取数据源统计"""
            return SourceStatistics(source_id=source_id)
        
        @self.app.get("/sources/{source_id}/health", response_model=SourceHealth)
        async def get_source_health(source_id: str):
            """获取数据源健康状态"""
            return SourceHealth(source_id=source_id)
        
        @self.app.get("/sources/types")
        async def get_supported_types():
            """获取支持的数据源类型"""
            return {
                "types": self.core.get_supported_source_types(),
                "info": {
                    "rss": {
                        "name": "RSS订阅",
                        "description": "RSS/Atom订阅源",
                        "required_config": ["url"],
                        "optional_config": ["update_interval", "max_items"]
                    },
                    "email": {
                        "name": "邮件订阅",
                        "description": "通过IMAP收取邮件",
                        "required_config": ["imap_server", "username", "password"],
                        "optional_config": ["folder", "max_emails", "filters"]
                    },
                    "crawler": {
                        "name": "网络爬虫",
                        "description": "网页内容爬取",
                        "required_config": ["url"],
                        "optional_config": ["selector", "headers", "cookies"]
                    }
                }
            }
        
        @self.app.get("/status")
        async def get_status():
            """获取服务状态"""
            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "version": "1.0.0",
                "supported_types": self.core.get_supported_source_types()
            }


# 创建API实例
knowbase_api = KnowBaseAPI()
app = knowbase_api.app