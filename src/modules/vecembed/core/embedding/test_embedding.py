"""
向量化SDK测试
"""

import asyncio
import pytest
import logging
from typing import List

from src.infrastructure.embedding import (
    VectorDatabaseConfig,
    EmbeddingConfig,
    EmbeddingManagerConfig,
    EmbeddingManager,
    DocumentInput,
    create_embedding_manager
)

# 设置测试日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEmbeddingSDK:
    """向量化SDK测试类"""
    
    @pytest.fixture
    async def manager(self):
        """创建测试用的嵌入管理器"""
        manager = await create_embedding_manager(
            vector_db_host="localhost",
            vector_db_port=6333,
            collection_name="test_collection",
            embedding_model="all-MiniLM-L6-v2"
        )
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """测试管理器初始化"""
        config = EmbeddingManagerConfig(
            vector_db_config=VectorDatabaseConfig(
                host="localhost",
                port=6333,
                collection_name="test_init"
            ),
            embedding_config=EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                model_type="sentence_transformers"
            )
        )
        
        manager = EmbeddingManager(config)
        
        try:
            success = await manager.initialize()
            assert success, "管理器初始化失败"
            assert manager._initialized, "管理器状态未正确设置"
        finally:
            await manager.close()
    
    @pytest.mark.asyncio
    async def test_add_and_search_documents(self, manager):
        """测试添加和搜索文档"""
        # 准备测试文档
        documents = [
            DocumentInput(
                content="人工智能是计算机科学的一个分支",
                metadata={"category": "AI", "language": "zh"}
            ),
            DocumentInput(
                content="机器学习是人工智能的子领域",
                metadata={"category": "ML", "language": "zh"}
            ),
            DocumentInput(
                content="深度学习使用神经网络",
                metadata={"category": "DL", "language": "zh"}
            )
        ]
        
        # 添加文档
        success = await manager.add_documents(documents)
        assert success, "添加文档失败"
        
        # 验证文档数量
        count = await manager.count_documents()
        assert count >= len(documents), f"文档数量不正确: {count}"
        
        # 搜索文档
        results = await manager.search_similar_documents("什么是AI", top_k=3)
        assert len(results) > 0, "搜索结果为空"
        assert results[0].score > 0, "相似度分数无效"
        
        # 验证搜索结果包含相关内容
        found_ai = any("人工智能" in result.document.content for result in results)
        assert found_ai, "搜索结果中未找到相关内容"
    
    @pytest.mark.asyncio
    async def test_document_chunking(self, manager):
        """测试文档分块功能"""
        # 创建长文档
        long_content = "人工智能技术发展。" * 200  # 创建一个长文档
        
        long_document = DocumentInput(
            content=long_content,
            metadata={"type": "long_document", "test": True}
        )
        
        # 添加长文档（启用分块）
        success = await manager.add_document(long_document, chunk_document=True)
        assert success, "添加长文档失败"
        
        # 搜索分块文档
        results = await manager.search_similar_documents("人工智能", top_k=5)
        
        # 验证是否有分块文档
        chunk_results = [r for r in results if "chunk_index" in r.document.metadata]
        assert len(chunk_results) > 0, "未找到分块文档"
    
    @pytest.mark.asyncio
    async def test_filtered_search(self, manager):
        """测试过滤搜索"""
        # 添加不同类别的文档
        documents = [
            DocumentInput(
                content="Python是一种编程语言",
                metadata={"category": "programming", "language": "zh"}
            ),
            DocumentInput(
                content="Java是面向对象的编程语言",
                metadata={"category": "programming", "language": "zh"}
            ),
            DocumentInput(
                content="机器学习算法很重要",
                metadata={"category": "AI", "language": "zh"}
            )
        ]
        
        await manager.add_documents(documents)
        
        # 过滤搜索
        results = await manager.search_similar_documents(
            "编程",
            top_k=5,
            filter_conditions={"category": "programming"}
        )
        
        # 验证过滤结果
        assert len(results) > 0, "过滤搜索结果为空"
        for result in results:
            assert result.document.metadata.get("category") == "programming", \
                "过滤条件未生效"
    
    @pytest.mark.asyncio
    async def test_document_operations(self, manager):
        """测试文档CRUD操作"""
        # 添加文档
        document = DocumentInput(
            id="test_doc_123",
            content="这是一个测试文档",
            metadata={"test": True}
        )
        
        success = await manager.add_document(document)
        assert success, "添加文档失败"
        
        # 获取文档
        retrieved_doc = await manager.get_document("test_doc_123")
        assert retrieved_doc is not None, "获取文档失败"
        assert retrieved_doc.content == document.content, "文档内容不匹配"
        
        # 更新文档
        updated_document = DocumentInput(
            id="test_doc_123",
            content="这是更新后的文档",
            metadata={"test": True, "updated": True}
        )
        
        success = await manager.update_document(updated_document)
        assert success, "更新文档失败"
        
        # 验证更新
        retrieved_doc = await manager.get_document("test_doc_123")
        assert "更新后" in retrieved_doc.content, "文档未正确更新"
        
        # 删除文档
        success = await manager.delete_document("test_doc_123")
        assert success, "删除文档失败"
        
        # 验证删除
        retrieved_doc = await manager.get_document("test_doc_123")
        assert retrieved_doc is None, "文档未正确删除"
    
    @pytest.mark.asyncio
    async def test_collection_info(self, manager):
        """测试集合信息获取"""
        # 添加一些文档
        documents = [
            DocumentInput(content=f"测试文档 {i}", metadata={"index": i})
            for i in range(5)
        ]
        
        await manager.add_documents(documents)
        
        # 获取集合信息
        info = await manager.get_collection_info()
        assert isinstance(info, dict), "集合信息格式错误"
        
        # 统计文档数量
        count = await manager.count_documents()
        assert count >= len(documents), f"文档数量统计错误: {count}"


async def run_manual_tests():
    """手动运行测试（不依赖pytest）"""
    print("开始手动测试...")
    
    try:
        # 创建管理器
        manager = await create_embedding_manager(
            collection_name="manual_test_collection"
        )
        
        print("✓ 管理器创建成功")
        
        # 测试添加文档
        documents = [
            DocumentInput(
                content="这是第一个测试文档",
                metadata={"test": "manual", "index": 1}
            ),
            DocumentInput(
                content="这是第二个测试文档",
                metadata={"test": "manual", "index": 2}
            )
        ]
        
        success = await manager.add_documents(documents)
        assert success
        print("✓ 文档添加成功")
        
        # 测试搜索
        results = await manager.search_similar_documents("测试文档", top_k=2)
        assert len(results) > 0
        print(f"✓ 搜索成功，找到 {len(results)} 个结果")
        
        # 测试统计
        count = await manager.count_documents()
        print(f"✓ 文档统计成功，总数: {count}")
        
        # 清理
        await manager.close()
        print("✓ 管理器关闭成功")
        
        print("\n所有手动测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行手动测试
    asyncio.run(run_manual_tests())