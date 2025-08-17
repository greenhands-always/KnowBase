import os
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Iterator, Dict, Any, Union
import json
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel

from src.infrastructure.utils import PathUtil,LLMUtil
# from src.infrastructure.utils import *
from src.infrastructure.utils.StreamUtil import (
    FileStreamBuilder, StreamProcessor, StreamItem,
    file_exists_filter, file_size_filter
)

# 尝试导入新版本的ChatOpenAI
try:
    from langchain_openai import ChatOpenAI as NewChatOpenAI
except ImportError:
    NewChatOpenAI = None

class ConceptExtraction(BaseModel):
    concepts: List[str] = Field(description="从文本中提取的关键概念列表")
    entities: List[str] = Field(description="重要的实体名称")
    keywords: List[str] = Field(description="核心关键词")


def extract_concepts_from_markdown_llm(file_path: str, llm) -> ConceptExtraction:
    """使用大模型从Markdown文件中提取概念"""
    # 更灵活的类型检查
    valid_types = (OpenAI, ChatOpenAI)
    if NewChatOpenAI:
        valid_types = valid_types + (NewChatOpenAI,)
    
    # 检查是否有invoke方法（Langchain LLM的基本特征）
    if not hasattr(llm, 'invoke') and not isinstance(llm, valid_types):
        raise ValueError(f"llm 参数必须是支持Langchain接口的LLM实例，当前类型: {type(llm)}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 设置输出解析器
        parser = PydanticOutputParser(pydantic_object=ConceptExtraction)

        # 创建提示模板
        prompt_template = PromptTemplate(
            template="""
            请从以下Markdown文档中提取关键概念、实体和关键词,并根据文章主题进对文章打标签,后续我需要对。
            请注意处理标点符号和文字之间的关系，准确识别完整的概念。

            文档内容：
            {content}

            {format_instructions}

            请确保：
            1. 提取完整的概念短语，不要被标点符号截断
            2. 识别专业术语和技术概念
            3. 包含重要的实体名称（人名、地名、组织名等）
            4. 提取核心关键词
            """,
            input_variables=["content"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # 创建链
        chain = prompt_template | llm | parser

        # 执行提取
        result = chain.invoke({"content": content})

        return result

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return ConceptExtraction(concepts=[], entities=[], keywords=[])


def concept_extraction_processor(llm):
    """
    创建概念提取处理器函数
    
    Args:
        llm: LLM实例
        
    Returns:
        处理器函数
    """
    def processor(item: StreamItem) -> ConceptExtraction:
        """概念提取处理器"""
        return extract_concepts_from_markdown_llm(item.file_path, llm)
    
    processor.__name__ = "concept_extraction"
    return processor


def process_files_with_stream(file_stream: Iterator[StreamItem], 
                            model,
                            progress_callback=None) -> List[Dict[str, Any]]:
    """
    使用流处理方式处理文件并提取概念
    
    Args:
        file_stream: 文件流
        model: LLM模型实例
        progress_callback: 进度回调函数
        
    Returns:
        处理结果列表，每个元素包含原始数据和概念提取结果
    """
    # 创建流处理器
    processor = StreamProcessor()
    processor.add_processor(concept_extraction_processor(model))
    
    # 处理流
    results = processor.process_stream(file_stream, progress_callback)
    
    # 转换结果格式
    processed_results = []
    for result in results:
        item = result['item']
        concepts = result['results'].get('concept_extraction')
        
        # 构建结果数据
        result_data = {
            'title': item.title,
            'file_path': str(item.file_path),
            'concepts': concepts
        }
        
        # 如果有JSON数据，则合并
        if 'json_data' in item.metadata:
            result_data.update(item.metadata['json_data'])
            result_data['concepts'] = concepts  # 确保concepts字段在最后
            
        processed_results.append(result_data)
    
    return processed_results


class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理Pydantic模型和其他特殊对象"""
    
    def default(self, obj):
        if isinstance(obj, BaseModel):
            # 对于Pydantic模型，转换为字典
            return obj.dict()
        elif isinstance(obj, Path):
            # 对于Path对象，转换为字符串
            return str(obj)
        elif isinstance(obj, datetime):
            # 对于datetime对象，转换为ISO格式字符串
            return obj.isoformat()
        return super().default(obj)


def save_results_to_file(results: List[Dict[str, Any]], output_file: Path):
    """
    保存结果到文件
    
    Args:
        results: 处理结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)


if __name__ == '__main__':
    base_dir = PathUtil.get_project_base_dir()
    result_base_dir = PathUtil.concat_path(base_dir, "result/Huggingface")
    
    # 只处理trending目录的数据
    trending_dir = PathUtil.concat_path(result_base_dir, "Blog/Trending")
    
    print("开始使用流处理方式处理trending目录的文件...")
    
    # 创建文件流构建器
    builder = FileStreamBuilder(trending_dir)
    
    # 构建文件流，添加过滤器确保文件存在且有内容
    file_stream = (builder
                   .add_filter(file_exists_filter)
                   .add_filter(file_size_filter(min_size=100))  # 过滤掉小于100字节的文件
                   .set_limit(1)
                   .build_from_directory("*.md"))
    
    # 定义进度回调函数
    def progress_callback(idx: int, title: str):
        print(f"正在处理第 {idx + 1} 个文件: {title}")
    
    provider = LLMUtil.CompatibleOpenAIProvider(os.getenv("ALIYUN_URL"),"ALIYUN_API_KEY")
    llm = provider.get_llm(model_name="qwen-turbo-latest")

    # provider = LLMUtil.OllamaProvider(model_name="zephyr")
    # llm = provider.get_llm(model_name="zephyr")
    # 处理文件流
    results = process_files_with_stream(
        file_stream=file_stream,
        model=llm,
        progress_callback=progress_callback
    )
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = PathUtil.concat_path(result_base_dir, f"trending_posts_with_concepts_stream_{timestamp}.json")
    
    # 保存结果
    save_results_to_file(results, output_file)
    
    print(f"\n流处理完成！")
    print(f"共处理了 {len(results)} 个文件")
    print(f"结果已保存到: {output_file}")
    
    # 显示处理结果统计
    successful_extractions = sum(1 for result in results if result.get('concepts') and 
                               (result['concepts'].concepts or result['concepts'].entities or result['concepts'].keywords))
    print(f"成功提取概念的文件数量: {successful_extractions}")
    

