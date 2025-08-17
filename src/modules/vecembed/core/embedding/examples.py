"""
向量化SDK使用示例
演示如何使用向量化SDK进行文档存储和搜索
"""

import asyncio
import logging
from typing import List

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入向量化SDK
from src.infrastructure.embedding import (
    create_embedding_manager,
    DocumentInput,
    EmbeddingManager,
    EmbeddingSystemConfig
)


async def basic_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 创建嵌入管理器（使用默认配置）
    manager = await create_embedding_manager(
        vector_db_host="localhost",
        vector_db_port=6333,
        collection_name="example_documents",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    try:
        # 准备示例文档
        documents = [
            DocumentInput(
                content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                metadata={"category": "AI", "language": "zh", "source": "example"}
            ),
            DocumentInput(
                content="机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
                metadata={"category": "ML", "language": "zh", "source": "example"}
            ),
            DocumentInput(
                content="深度学习是机器学习的一个分支，使用神经网络来模拟人脑的学习过程。",
                metadata={"category": "DL", "language": "zh", "source": "example"}
            ),
            DocumentInput(
                content="Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
                metadata={"category": "NLP", "language": "en", "source": "example"}
            )
        ]
        
        # 添加文档到向量数据库
        print("添加文档到向量数据库...")
        success = await manager.add_documents(documents)
        if success:
            print(f"成功添加 {len(documents)} 个文档")
        else:
            print("添加文档失败")
            return
        
        # 查看集合信息
        info = await manager.get_collection_info()
        print(f"集合信息: {info}")
        
        # 统计文档数量
        count = await manager.count_documents()
        print(f"文档总数: {count}")
        
        # 搜索相似文档
        print("\n搜索相似文档...")
        query = "什么是人工智能？"
        results = await manager.search_similar_documents(query, top_k=3)
        
        print(f"查询: '{query}'")
        print(f"找到 {len(results)} 个相关文档:")
        for i, result in enumerate(results, 1):
            print(f"{i}. 相似度: {result.score:.4f}")
            print(f"   内容: {result.document.content[:100]}...")
            print(f"   元数据: {result.document.metadata}")
            print()
        
        # 使用过滤条件搜索
        print("使用过滤条件搜索...")
        filtered_results = await manager.search_similar_documents(
            "machine learning algorithms",
            top_k=5,
            filter_conditions={"language": "zh"}
        )
        
        print(f"过滤结果 (language=zh): {len(filtered_results)} 个文档")
        for result in filtered_results:
            print(f"- {result.document.content[:50]}... (相似度: {result.score:.4f})")
        
    finally:
        # 关闭管理器
        await manager.close()


async def advanced_example():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")
    
    # 使用配置文件创建管理器
    config = EmbeddingSystemConfig.from_env()
    manager_config = config.create_manager_config(
        collection_name="advanced_documents",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    manager = EmbeddingManager(manager_config)
    await manager.initialize()
    
    try:
        # 添加长文档（会自动分块）
        long_document = DocumentInput(
            content="""
            人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
            这些任务包括学习、推理、问题解决、感知和语言理解。AI的发展历史可以追溯到20世纪50年代，当时科学家们开始探索
            如何让机器模拟人类的思维过程。
            
            机器学习是人工智能的一个重要子领域，专注于开发能够从数据中学习和改进的算法。与传统的编程方法不同，
            机器学习算法不需要明确的指令来执行特定任务，而是通过分析大量数据来识别模式和做出预测。
            
            深度学习是机器学习的一个分支，使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、语音识别、
            自然语言处理等领域取得了突破性进展，推动了AI技术的快速发展。
            
            自然语言处理（NLP）是AI的另一个重要分支，专注于计算机与人类语言之间的交互。NLP技术使计算机能够理解、
            解释和生成人类语言，为聊天机器人、机器翻译、文本摘要等应用提供了基础。
            """,
            metadata={
                "title": "人工智能概述",
                "category": "AI",
                "author": "示例作者",
                "language": "zh"
            }
        )
        
        print("添加长文档（自动分块）...")
        await manager.add_document(long_document, chunk_document=True)
        
        # 搜索并显示分块结果
        results = await manager.search_similar_documents("深度学习神经网络", top_k=5)
        print(f"搜索结果: {len(results)} 个文档块")
        
        for i, result in enumerate(results, 1):
            metadata = result.document.metadata
            chunk_info = ""
            if "chunk_index" in metadata:
                chunk_info = f" (块 {metadata['chunk_index'] + 1}/{metadata['total_chunks']})"
            
            print(f"{i}. 相似度: {result.score:.4f}{chunk_info}")
            print(f"   内容: {result.document.content[:100]}...")
            print()
        
        # 获取特定文档
        if results:
            doc_id = results[0].document.id
            document = await manager.get_document(doc_id)
            if document:
                print(f"获取文档 {doc_id}:")
                print(f"内容长度: {len(document.content)}")
                print(f"元数据: {document.metadata}")
        
    finally:
        await manager.close()


async def openai_example():
    """OpenAI向量化示例（需要API密钥）"""
    print("\n=== OpenAI向量化示例 ===")
    
    # 检查是否有OpenAI API密钥
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("跳过OpenAI示例：未设置OPENAI_API_KEY环境变量")
        return
    
    # 使用OpenAI配置
    config = EmbeddingSystemConfig(
        default_embedding_model="text-embedding-3-small",
        default_embedding_type="openai",
        openai_api_key=api_key
    )
    
    manager_config = config.create_manager_config(collection_name="openai_documents")
    manager = EmbeddingManager(manager_config)
    
    try:
        await manager.initialize()
        
        # 添加文档
        documents = [
            DocumentInput(
                content="OpenAI's GPT models are large language models trained on diverse text data.",
                metadata={"model": "GPT", "company": "OpenAI"}
            ),
            DocumentInput(
                content="DALL-E is an AI system that can create realistic images from text descriptions.",
                metadata={"model": "DALL-E", "company": "OpenAI"}
            )
        ]
        
        await manager.add_documents(documents)
        
        # 搜索
        results = await manager.search_similar_documents("image generation AI", top_k=2)
        print(f"OpenAI向量化搜索结果: {len(results)} 个文档")
        
        for result in results:
            print(f"- 相似度: {result.score:.4f}")
            print(f"  内容: {result.document.content}")
            print()
            
    except Exception as e:
        print(f"OpenAI示例出错: {e}")
    finally:
        await manager.close()


async def main():
    """主函数"""
    print("向量化SDK使用示例")
    print("=" * 50)
    
    try:
        # 运行基础示例
        await basic_example()
        
        # 运行高级示例
        await advanced_example()
        
        # 运行OpenAI示例
        await openai_example()
        
    except Exception as e:
        logger.error(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())