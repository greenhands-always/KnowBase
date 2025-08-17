"""
VecEmbed (向量工坊) - 多模态信息向量化引擎

提供文本向量化服务，支持多种向量化模型：
- Sentence Transformers
- OpenAI Embeddings
- Hugging Face Transformers
- 本地模型
"""

from .core import VecEmbedCore
from .api import VecEmbedAPI
from .models import EmbeddingRequest, EmbeddingResponse, SearchRequest, SearchResponse

__all__ = ["VecEmbedCore", "VecEmbedAPI", "EmbeddingRequest", "EmbeddingResponse", "SearchRequest", "SearchResponse"]