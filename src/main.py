#!/usr/bin/env python3
"""
PKM Copilot 主应用
统一的个人知识管理系统入口
"""

import asyncio
import logging
import argparse
from pathlib import Path
from typing import Optional

from src.core.config import get_config, create_default_config
from src.modules.knowbase.core import KnowBaseCore
from src.modules.vecembed.core import VecEmbedCore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PkmCopilot:
    """PKM Copilot主应用类"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.knowbase = KnowBaseCore()
        self.vecembed = VecEmbedCore()
        self._initialized = False
    
    async def initialize(self):
        """初始化应用"""
        if self._initialized:
            return
        
        logger.info("正在初始化PKM Copilot...")
        
        # 初始化向量存储
        if self.config.vecembed.enabled:
            vector_size = self.vecembed.get_vector_size()
            await self.vecembed.create_collection(vector_size)
            logger.info(f"向量存储已初始化，维度: {vector_size}")
        
        self._initialized = True
        logger.info("PKM Copilot初始化完成")
    
    async def run_workflow(self, workflow_type: str = "full"):
        """运行工作流"""
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"开始运行工作流: {workflow_type}")
        
        if workflow_type == "collect":
            await self._run_collection_workflow()
        elif workflow_type == "embed":
            await self._run_embedding_workflow()
        elif workflow_type == "full":
            await self._run_full_workflow()
        else:
            logger.error(f"不支持的工作流类型: {workflow_type}")
    
    async def _run_collection_workflow(self):
        """运行收集工作流"""
        logger.info("运行收集工作流...")
        # 这里实现数据源收集逻辑
        
    async def _run_embedding_workflow(self):
        """运行向量化工作流"""
        logger.info("运行向量化工作流...")
        # 这里实现内容向量化逻辑
        
    async def _run_full_workflow(self):
        """运行完整工作流"""
        logger.info("运行完整工作流...")
        await self._run_collection_workflow()
        await self._run_embedding_workflow()
    
    def get_status(self) -> dict:
        """获取应用状态"""
        return {
            "initialized": self._initialized,
            "config": {
                "knowbase_enabled": self.config.knowbase.enabled,
                "vecembed_enabled": self.config.vecembed.enabled,
            },
            "modules": {
                "knowbase": {
                    "supported_types": self.knowbase.get_supported_source_types()
                },
                "vecembed": {
                    "available_models": self.vecembed.get_available_models()
                }
            }
        }


class PkmCopilotCLI:
    """PKM Copilot命令行接口"""
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="PKM Copilot - 个人知识管理系统")
        self._setup_commands()
    
    def _setup_commands(self):
        """设置命令行参数"""
        subparsers = self.parser.add_subparsers(dest='command', help='可用命令')
        
        # 启动命令
        start_parser = subparsers.add_parser('start', help='启动PKM Copilot')
        start_parser.add_argument('--config', help='配置文件路径')
        start_parser.add_argument('--workflow', choices=['collect', 'embed', 'full'], 
                                default='full', help='工作流类型')
        
        # 初始化命令
        init_parser = subparsers.add_parser('init', help='初始化配置')
        init_parser.add_argument('--template', choices=['development', 'production', 'test'],
                               default='development', help='配置模板')
        init_parser.add_argument('--output', default='config/pkm_copilot.yaml', help='输出路径')
        
        # 状态命令
        status_parser = subparsers.add_parser('status', help='查看系统状态')
        status_parser.add_argument('--config', help='配置文件路径')
        
        # 模块命令
        module_parser = subparsers.add_parser('module', help='模块管理')
        module_subparsers = module_parser.add_subparsers(dest='module_action')
        
        # 列出模块
        list_parser = module_subparsers.add_parser('list', help='列出所有模块')
        
        # 测试模块
        test_parser = module_subparsers.add_parser('test', help='测试模块')
        test_parser.add_argument('module_name', choices=['knowbase', 'vecembed'], help='模块名称')
    
    async def run(self):
        """运行命令行"""
        args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        if args.command == 'start':
            await self._cmd_start(args)
        elif args.command == 'init':
            await self._cmd_init(args)
        elif args.command == 'status':
            await self._cmd_status(args)
        elif args.command == 'module':
            await self._cmd_module(args)
    
    async def _cmd_start(self, args):
        """处理启动命令"""
        app = PkmCopilot(args.config)
        await app.run_workflow(args.workflow)
    
    async def _cmd_init(self, args):
        """处理初始化命令"""
        from src.core.config import create_config_template
        success = create_config_template(args.template, args.output)
        if success:
            print(f"配置已初始化: {args.output}")
        else:
            print("配置初始化失败")
    
    async def _cmd_status(self, args):
        """处理状态命令"""
        app = PkmCopilot(args.config)
        status = app.get_status()
        
        print("PKM Copilot 状态:")
        print(f"  已初始化: {status['initialized']}")
        print(f"  模块状态:")
        for module, info in status['modules'].items():
            print(f"    {module}: {info}")
    
    async def _cmd_module(self, args):
        """处理模块命令"""
        if args.module_action == 'list':
            print("可用模块:")
            print("  - knowbase: 多源信息聚合中枢")
            print("  - vecembed: 多模态信息向量化引擎")
            print("  - siftflow: AI智能过滤系统 (开发中)")
            print("  - sumagent: 自动化内容提炼工具 (开发中)")
            print("  - linkverse: 可视化知识图谱模块 (开发中)")
            print("  - collectdeck: 个人知识交互入口 (开发中)")
        
        elif args.module_action == 'test':
            print(f"正在测试模块: {args.module_name}")
            app = PkmCopilot()
            await app.initialize()
            
            if args.module_name == 'knowbase':
                print("KnowBase模块测试:")
                print(f"  支持的数据源类型: {app.knowbase.get_supported_source_types()}")
            
            elif args.module_name == 'vecembed':
                print("VecEmbed模块测试:")
                models = app.vecembed.get_available_models()
                print(f"  可用模型: {models}")


async def main():
    """主函数"""
    cli = PkmCopilotCLI()
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())