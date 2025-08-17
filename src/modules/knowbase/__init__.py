"""
KnowBase (知库) - 多源信息聚合中枢

提供统一的数据收集接口，支持多种数据源：
- RSS订阅
- 邮件订阅
- 网络爬虫
- API接口
- 手动输入
- 文件导入
"""

from .core import KnowBaseCore
from .api import KnowBaseAPI
from .models import SourceConfig, SyncStatus, DataSource

__all__ = ["KnowBaseCore", "KnowBaseAPI", "SourceConfig", "SyncStatus", "DataSource"]