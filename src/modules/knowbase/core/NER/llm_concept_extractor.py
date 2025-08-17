import os
import csv
import json
import time
from typing import List, Dict, Any
from collections import defaultdict
import requests
from openai import OpenAI
import re


class LLMConceptExtractor:
    """基于大模型的概念提取器"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", base_url: str = None):
        """
        初始化大模型概念提取器
        
        Args:
            api_key: API密钥
            model: 使用的模型名称
            base_url: API基础URL
        """
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
    def extract_concepts_from_markdown(self, file_path: str) -> List[str]:
        """
        从Markdown文件中提取概念
        
        Args:
            file_path: Markdown文件路径
            
        Returns:
            提取的概念列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 清理内容
            cleaned_content = self._clean_markdown(content)
            
            # 使用大模型提取概念
            concepts = self._extract_concepts_with_llm(cleaned_content)
            
            return concepts
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            return []
    
    def _clean_markdown(self, content: str) -> str:
        """
        清理Markdown格式
        
        Args:
            content: 原始内容
            
        Returns:
            清理后的内容
        """
        # 移除Markdown标记
        content = re.sub(r'#+\s*', '', content)  # 移除标题标记
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # 移除粗体
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # 移除斜体
        content = re.sub(r'`([^`]+)`', r'\1', content)  # 移除行内代码
        content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)  # 移除代码块
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # 移除图片
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # 移除链接，保留文本
        content = re.sub(r'^\s*[-*+]\s*', '', content, flags=re.MULTILINE)  # 移除列表标记
        content = re.sub(r'\n{3,}', '\n\n', content)  # 规范化空行
        
        return content.strip()
    
    def _extract_concepts_with_llm(self, content: str) -> List[str]:
        """
        使用大模型提取概念
        
        Args:
            content: 清理后的内容
            
        Returns:
            提取的概念列表
        """
        if len(content) > 8000:
            content = content[:8000] + "..."
            
        prompt = """
        请从以下技术文章中提取重要的技术概念、术语、框架名称、库名称、算法名称等。
        要求：
        1. 提取的概念应该是有意义的技术术语
        2. 排除过于通用的词汇
        3. 保留专有名词和技术术语
        4. 返回JSON格式的概念列表
        
        文章内容：
        {content}
        
        请返回以下格式的JSON：
        {{
            "concepts": ["概念1", "概念2", "概念3", ...]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的技术概念提取助手，擅长从文章中提取精确的技术术语和概念。"},
                    {"role": "user", "content": prompt.format(content=content)}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content
            
            # 解析JSON响应
            try:
                json_match = re.search(r'\{[^{}]*"concepts"[^{}]*\}', result, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    concepts = data.get("concepts", [])
                    return [str(concept).strip() for concept in concepts if concept.strip()]
                else:
                    # 尝试直接提取列表
                    list_match = re.search(r'\[(.*?)\]', result, re.DOTALL)
                    if list_match:
                        items = re.findall(r'"([^"]+)"', list_match.group())
                        return [item.strip() for item in items if item.strip()]
            except json.JSONDecodeError:
                print("解析JSON响应失败")
                return []
                
        except Exception as e:
            print(f"调用大模型API时出错: {e}")
            return []
        
        return []
    
    def process_directory(self, directory: str, limit: int = 30) -> Dict[str, int]:
        """
        处理目录下的所有Markdown文件
        
        Args:
            directory: 目录路径
            limit: 处理的文件数量限制
            
        Returns:
            概念频率字典
        """
        concept_freq = defaultdict(int)
        cnt = 0
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    print(f"正在处理文件: {file_path}")
                    cnt += 1
                    
                    if cnt > limit:
                        print("已达到处理文件数量限制，停止处理。")
                        break
                        
                    concepts = self.extract_concepts_from_markdown(file_path)
                    
                    # 统计词频
                    for concept in concepts:
                        concept_freq[concept] += 1
                        
                    # 避免API调用过于频繁
                    time.sleep(0.5)
        
        return concept_freq
    
    def save_concept_frequency(self, concept_freq: Dict[str, int], output_file: str):
        """
        将概念频率保存到CSV文件
        
        Args:
            concept_freq: 概念频率字典
            output_file: 输出文件路径
        """
        # 按频率降序排序
        sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 确保输出文件扩展名为.csv
        if not output_file.endswith('.csv'):
            output_file = output_file.replace('.txt', '.csv')
            if not output_file.endswith('.csv'):
                output_file = output_file + '.csv'
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["概念", "频率"])
            for concept, freq in sorted_concepts:
                writer.writerow([str(concept), freq])
        
        print(f"概念频率已保存到 {output_file}")
    
    def save_detailed_results(self, concept_freq: Dict[str, int], output_file: str):
        """
        保存详细的分析结果，包括额外的统计信息
        
        Args:
            concept_freq: 概念频率字典
            output_file: 输出文件路径
        """
        sorted_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 计算统计信息
        total_concepts = sum(concept_freq.values())
        unique_concepts = len(concept_freq)
        
        detailed_output = output_file.replace('.csv', '_detailed.json')
        
        with open(detailed_output, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": {
                    "total_occurrences": total_concepts,
                    "unique_concepts": unique_concepts,
                    "average_frequency": total_concepts / unique_concepts if unique_concepts > 0 else 0
                },
                "concepts": [
                    {
                        "concept": concept,
                        "frequency": freq,
                        "percentage": round(freq / total_concepts * 100, 2) if total_concepts > 0 else 0
                    }
                    for concept, freq in sorted_concepts[:100]  # 只保留前100个
                ]
            }, f, ensure_ascii=False, indent=2)
        
        print(f"详细分析结果已保存到 {detailed_output}")


def main():
    """主函数示例"""
    import sys
    
    # 设置API密钥
    api_key = os.getenv("OPENAI_API_KEY") or "your-api-key-here"
    
    # 初始化提取器
    extractor = LLMConceptExtractor(
        api_key=api_key,
        model="gpt-4o-mini"
    )
    
    # 设置目录路径
    directory = r"D:\code\ai-trend-summary\result\Huggingface\Blog\Trending"
    
    # 输出文件路径
    output_file = r"D:\code\ai-trend-summary\result\Huggingface\Blog\llm_concept_frequency.csv"
    
    # 处理目录并统计概念频率
    concept_freq = extractor.process_directory(directory, limit=10)  # 先处理少量文件测试
    
    # 保存结果
    extractor.save_concept_frequency(concept_freq, output_file)
    extractor.save_detailed_results(concept_freq, output_file)


if __name__ == '__main__':
    main()