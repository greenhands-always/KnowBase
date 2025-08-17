import os
import csv
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict, Counter
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from llm_client import LLMClient
from config import Config


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedConceptExtractor:
    """增强版概念提取器，支持多种LLM提供商"""
    
    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        rate_limit: float = 1.0,
        max_workers: int = 1
    ):
        """
        初始化增强版概念提取器
        
        Args:
            provider: LLM提供商 (openai, anthropic, gemini, ollama)
            model: 模型名称
            api_key: API密钥
            base_url: API基础URL (对于Ollama)
            rate_limit: API调用间隔(秒)
            max_workers: 最大并发数
        """
        self.provider = provider
        self.rate_limit = rate_limit
        self.max_workers = max_workers
        
        # 自动获取API密钥
        if not api_key:
            api_key = self._get_api_key(provider)
        
        # 设置默认模型
        if not model:
            model = self._get_default_model(provider)
        
        self.llm_client = LLMClient(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url
        )
        
        # 缓存已处理文件的结果，避免重复调用
        self.cache = {}
        
    def _get_api_key(self, provider: str) -> Optional[str]:
        """获取API密钥"""
        key_map = {
            "openai": Config.get_openai_api_key(),
            "anthropic": Config.get_anthropic_api_key(),
            "gemini": Config.get_gemini_api_key(),
            "ollama": "ollama"  # Ollama不需要密钥
        }
        
        key = key_map.get(provider)
        if not key:
            logger.warning(f"未找到 {provider} 的API密钥，请设置环境变量")
        return key
    
    def _get_default_model(self, provider: str) -> str:
        """获取默认模型"""
        models = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "gemini": "gemini-pro",
            "ollama": "llama3.1"
        }
        return models.get(provider, "gpt-4o-mini")
    
    def _clean_markdown(self, content: str) -> str:
        """
        清理Markdown格式，保留语义内容
        
        Args:
            content: 原始Markdown内容
            
        Returns:
            清理后的纯文本内容
        """
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        
        # 移除Markdown标记，但保留内容
        content = re.sub(r'#+\s*([^\n]+)', r'\1', content)  # 标题
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # 粗体
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # 斜体
        content = re.sub(r'`([^`]+)`', r'\1', content)  # 行内代码
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)  # 代码块
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # 图片
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # 链接保留文本
        content = re.sub(r'^\s*[-*+]\s*', '', content, flags=re.MULTILINE)  # 无序列表
        content = re.sub(r'^\s*\d+\.\s*', '', content, flags=re.MULTILINE)  # 有序列表
        content = re.sub(r'^\s*>\s*', '', content, flags=re.MULTILINE)  # 引用
        content = re.sub(r'\n{3,}', '\n\n', content)  # 规范化空行
        content = re.sub(r'\|', ' ', content)  # 表格分隔符
        content = re.sub(r'-{3,}', '', content)  # 分隔线
        
        # 移除多余空格
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _split_content(self, content: str, max_length: int = 4000) -> List[str]:
        """
        将长内容分割成多个片段
        
        Args:
            content: 原始内容
            max_length: 每个片段的最大长度
            
        Returns:
            内容片段列表
        """
        if len(content) <= max_length:
            return [content]
        
        # 按句子分割
        sentences = re.split(r'[.!?]+', content)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_concepts_from_file(self, file_path: str, use_cache: bool = True) -> List[str]:
        """
        从单个文件中提取概念
        
        Args:
            file_path: 文件路径
            use_cache: 是否使用缓存
            
        Returns:
            概念列表
        """
        if use_cache and file_path in self.cache:
            return self.cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 清理内容
            cleaned_content = self._clean_markdown(content)
            
            # 分割长内容
            chunks = self._split_content(cleaned_content)
            
            # 从每个片段提取概念
            all_concepts = []
            for chunk in chunks:
                if chunk.strip():
                    concepts = self.llm_client.extract_concepts(chunk)
                    all_concepts.extend(concepts)
                    time.sleep(self.rate_limit)
            
            # 去重并保持顺序
            seen = set()
            unique_concepts = []
            for concept in all_concepts:
                if concept.lower() not in seen:
                    seen.add(concept.lower())
                    unique_concepts.append(concept)
            
            # 过滤无效概念
            filtered_concepts = self._filter_concepts(unique_concepts)
            
            if use_cache:
                self.cache[file_path] = filtered_concepts
            
            return filtered_concepts
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            return []
    
    def _filter_concepts(self, concepts: List[str]) -> List[str]:
        """
        过滤无效概念
        
        Args:
            concepts: 原始概念列表
            
        Returns:
            过滤后的概念列表
        """
        # 过滤规则
        stop_words = {
            '技术', '应用', '数据', '信息', '系统', '方法', '工具', '平台',
            '服务', '功能', '特性', '优势', '缺点', '问题', '解决方案',
            '开发', '实现', '使用', '需要', '可以', '能够', '进行',
            '提供', '支持', '包括', '包含', '例如', '比如', '等', '等等',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'without', 'toward', 'against', 'upon', 'about', 'until', 'while'
        }
        
        filtered = []
        for concept in concepts:
            concept = concept.strip()
            if not concept:
                continue
            
            # 跳过太短或太长的概念
            if len(concept) < 2 or len(concept) > 50:
                continue
            
            # 跳过纯数字
            if concept.isdigit():
                continue
            
            # 跳过常见停用词
            if concept.lower() in stop_words:
                continue
            
            # 跳过包含特殊字符的概念
            if re.search(r'[^\w\s\-_.]', concept):
                continue
            
            filtered.append(concept)
        
        return filtered
    
    def process_directory(
        self,
        directory: str,
        limit: int = None,
        file_pattern: str = "*.md",
        parallel: bool = False
    ) -> Dict[str, int]:
        """
        处理目录下的所有文件
        
        Args:
            directory: 目录路径
            limit: 处理的文件数量限制
            file_pattern: 文件匹配模式
            parallel: 是否并行处理
            
        Returns:
            概念频率字典
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"目录不存在: {directory}")
            return {}
        
        # 获取所有匹配的文件
        files = list(directory_path.rglob(file_pattern))
        if limit:
            files = files[:limit]
        
        logger.info(f"找到 {len(files)} 个文件待处理")
        
        concept_freq = defaultdict(int)
        
        if parallel and self.max_workers > 1:
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.extract_concepts_from_file, str(file_path)): file_path
                    for file_path in files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        concepts = future.result()
                        for concept in concepts:
                            concept_freq[concept] += 1
                        logger.info(f"处理完成: {file_path.name} - 提取了 {len(concepts)} 个概念")
                    except Exception as e:
                        logger.error(f"处理文件 {file_path} 时出错: {e}")
        else:
            # 串行处理
            for file_path in files:
                concepts = self.extract_concepts_from_file(str(file_path))
                for concept in concepts:
                    concept_freq[concept] += 1
                logger.info(f"处理完成: {file_path.name} - 提取了 {len(concepts)} 个概念")
        
        return dict(concept_freq)
    
    def save_results(
        self,
        concept_freq: Dict[str, int],
        output_base: str,
        include_details: bool = True
    ):
        """
        保存分析结果
        
        Args:
            concept_freq: 概念频率字典
            output_base: 输出文件基础名称
            include_details: 是否包含详细信息
        """
        # 按频率排序
        sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 保存CSV格式
        csv_file = f"{output_base}.csv"
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["概念", "频率", "占比"])
            
            total = sum(concept_freq.values())
            for concept, freq in sorted_concepts:
                percentage = round(freq / total * 100, 2) if total > 0 else 0
                writer.writerow([concept, freq, f"{percentage}%"])
        
        logger.info(f"CSV结果已保存到: {csv_file}")
        
        if include_details:
            # 保存JSON格式详细信息
            json_file = f"{output_base}_details.json"
            details = {
                "metadata": {
                    "provider": self.provider,
                    "total_concepts": len(concept_freq),
                    "total_occurrences": sum(concept_freq.values()),
                    "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                },
                "concepts": [
                    {
                        "concept": concept,
                        "frequency": freq,
                        "rank": i + 1
                    }
                    for i, (concept, freq) in enumerate(sorted_concepts)
                ],
                "statistics": {
                    "top_10_concepts": sorted_concepts[:10],
                    "concepts_by_frequency": dict(Counter([freq for _, freq in sorted_concepts]))
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(details, f, ensure_ascii=False, indent=2)
            
            logger.info(f"详细结果已保存到: {json_file}")
    
    def generate_summary_report(
        self,
        concept_freq: Dict[str, int],
        output_file: str,
        top_k: int = 20
    ):
        """
        生成概念分析摘要报告
        
        Args:
            concept_freq: 概念频率字典
            output_file: 输出文件路径
            top_k: 显示前K个概念
        """
        sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)
        
        total = sum(concept_freq.values())
        unique = len(concept_freq)
        
        report = f"""
# 技术概念分析报告

## 统计摘要
- **总概念数**: {unique}
- **总出现次数**: {total}
- **平均频率**: {total/unique:.2f}
- **分析时间**: {time.strftime("%Y-%m-%d %H:%M:%S")}
- **使用模型**: {self.provider}

## 热门概念 (Top {top_k})

"""
        
        for i, (concept, freq) in enumerate(sorted_concepts[:top_k], 1):
            percentage = freq / total * 100
            report += f"{i}. **{concept}** - 出现 {freq} 次 ({percentage:.1f}%)\n"
        
        report += f"""
## 概念分布
- **高频概念** (出现5次以上): {sum(1 for _, f in sorted_concepts if f >= 5)}
- **中频概念** (出现2-4次): {sum(1 for _, f in sorted_concepts if 2 <= f <= 4)}
- **低频概念** (出现1次): {sum(1 for _, f in sorted_concepts if f == 1)}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report.strip())
        
        logger.info(f"摘要报告已保存到: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="使用大模型提取技术概念")
    parser.add_argument("--directory", type=str, default=r"D:\code\ai-trend-summary\result\Huggingface\Blog\Trending")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "anthropic", "gemini", "ollama"])
    parser.add_argument("--model", type=str, help="模型名称")
    parser.add_argument("--api-key", type=str, help="API密钥")
    parser.add_argument("--limit", type=int, help="处理文件数量限制")
    parser.add_argument("--output", type=str, help="输出文件基础名称")
    parser.add_argument("--parallel", action="store_true", help="并行处理")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="API调用间隔(秒)")
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = EnhancedConceptExtractor(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        rate_limit=args.rate_limit
    )
    
    # 处理目录
    concept_freq = extractor.process_directory(
        args.directory,
        limit=args.limit,
        parallel=args.parallel
    )
    
    # 设置输出文件名
    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"enhanced_concepts_{args.provider}_{timestamp}"
    
    # 保存结果
    extractor.save_results(concept_freq, args.output)
    extractor.generate_summary_report(
        concept_freq,
        f"{args.output}_summary.md"
    )
    
    print(f"\n处理完成！")
    print(f"提取了 {len(concept_freq)} 个独特概念")
    print(f"总出现次数: {sum(concept_freq.values())}")
    print(f"结果文件: {args.output}.csv")


if __name__ == "__main__":
    main()