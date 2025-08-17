"""
VecEmbed API接口
提供向量化和搜索的RESTful API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime

from .core import VecEmbedCore
from .models import (
    EmbeddingRequest, EmbeddingResponse, BatchEmbeddingRequest, BatchEmbeddingResponse,
    SearchRequest, SearchResponse, VectorDataRequest, VectorStorageRequest,
    CollectionCreateRequest, CollectionInfo, ModelInfo, HealthStatus
)


class VecEmbedAPI:
    """VecEmbed API实现"""
    
    def __init__(self):
        self.core = VecEmbedCore()
        self.app = FastAPI(
            title="VecEmbed API",
            description="多模态信息向量化引擎",
            version="1.0.0"
        )
        self._setup_routes()
    
    def _setup_routes(self):
        """设置API路由"""
        
        @self.app.post("/embed", response_model=EmbeddingResponse)
        async def embed_text(request: EmbeddingRequest):
            """单文本向量化"""
            try:
                start_time = datetime.now()
                embedding = await self.core.vectorize_text(
                    request.text,
                    model_name=request.model_name
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return EmbeddingResponse(
                    embedding=embedding,
                    model_name=request.model_name or self.core.config.embedding_model,
                    provider="sentence_transformers",
                    vector_size=len(embedding),
                    processing_time=processing_time
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/embed/batch", response_model=BatchEmbeddingResponse)
        async def embed_batch(request: BatchEmbeddingRequest):
            """批量文本向量化"""
            try:
                start_time = datetime.now()
                embeddings = await self.core.vectorize_batch(
                    request.texts,
                    model_name=request.model_name
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return BatchEmbeddingResponse(
                    embeddings=embeddings,
                    model_name=request.model_name or self.core.config.embedding_model,
                    provider="sentence_transformers",
                    vector_size=len(embeddings[0]) if embeddings else 0,
                    processing_time=processing_time,
                    batch_size=len(request.texts)
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/search", response_model=SearchResponse)
        async def search_similar(request: SearchRequest):
            """语义搜索"""
            try:
                start_time = datetime.now()
                results = await self.core.search_similar(
                    request.query,
                    limit=request.limit
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 格式化结果
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "id": result.id,
                        "article_id": result.article_id,
                        "content": result.content,
                        "score": 0.0,  # 需要实现相似度计算
                        "metadata": result.metadata
                    })
                
                return SearchResponse(
                    results=formatted_results,
                    query=request.query,
                    total_count=len(formatted_results),
                    processing_time=processing_time,
                    search_type=request.search_type
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/vectors/store")
        async def store_vectors(request: VectorStorageRequest):
            """存储向量"""
            try:
                # 这里应该实现实际的向量存储逻辑
                # 目前返回成功状态
                return {
                    "success": True,
                    "stored_count": len(request.vectors),
                    "collection": request.collection_name
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/collections", response_model=Dict[str, Any])
        async def create_collection(request: CollectionCreateRequest):
            """创建向量集合"""
            try:
                success = await self.core.create_collection(
                    vector_size=request.vector_size
                )
                
                return {
                    "success": success,
                    "name": request.name,
                    "vector_size": request.vector_size,
                    "distance_metric": request.distance_metric
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/collections")
        async def list_collections():
            """列出所有向量集合"""
            try:
                # 这里应该实现实际的集合列表获取
                return {
                    "collections": [
                        {
                            "name": "pkm_articles",
                            "vector_size": 384,
                            "points_count": 0,
                            "created_at": datetime.now()
                        }
                    ]
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models")
        async def get_available_models():
            """获取可用模型"""
            try:
                models = await self.core.get_available_models()
                
                model_info = []
                for provider, model_list in models.items():
                    for model in model_list:
                        info = ModelInfo(
                            name=model,
                            provider=provider,
                            vector_size=self.core.get_vector_size(model),
                            max_tokens=512,
                            supported_languages=["zh", "en"],
                            description=f"{provider} embedding model",
                            is_available=True
                        )
                        model_info.append(info)
                
                return {"models": model_info}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/vectors/article/{article_id}")
        async def delete_article_vectors(article_id: str):
            """删除文章的向量"""
            try:
                success = await self.core.delete_article_vectors(article_id)
                return {"success": success, "article_id": article_id}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=HealthStatus)
        async def health_check():
            """健康检查"""
            try:
                return HealthStatus(
                    provider="sentence_transformers",
                    model_name=self.core.config.embedding_model,
                    vector_store=get_config().vector_store.provider,
                    collections=1,
                    total_vectors=0,
                    uptime=0.0
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/status")
        async def get_status():
            """获取服务状态"""
            try:
                return {
                    "status": "healthy",
                    "timestamp": datetime.now(),
                    "version": "1.0.0",
                    "models": await self.core.get_available_models(),
                    "config": {
                        "embedding_model": self.core.config.embedding_model,
                        "vector_store": get_config().vector_store.provider
                    }
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))


# 创建API实例
vecembed_api = VecEmbedAPI()
app = vecembed_api.app