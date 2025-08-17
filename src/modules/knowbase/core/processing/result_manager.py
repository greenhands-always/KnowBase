"""
结果管理器模块
提供处理结果的存储、加载和格式转换功能
"""

import json
import csv
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

from .article_processor import ProcessingResult


class ResultFormat(str, Enum):
    """结果格式枚举"""
    JSON = "json"
    CSV = "csv"
    JSONL = "jsonl"  # JSON Lines格式
    EXCEL = "excel"


class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理特殊对象"""
    
    def default(self, obj):
        if isinstance(obj, BaseModel):
            return obj.dict()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class ResultManager:
    """处理结果管理器"""
    
    def __init__(self):
        self.encoder = CustomJSONEncoder
    
    def save_results(self, 
                    results: List[ProcessingResult], 
                    output_path: Union[str, Path],
                    format: ResultFormat = ResultFormat.JSON,
                    **kwargs) -> None:
        """
        保存处理结果
        
        Args:
            results: 处理结果列表
            output_path: 输出路径
            format: 输出格式
            **kwargs: 额外参数
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == ResultFormat.JSON:
            self._save_as_json(results, output_path, **kwargs)
        elif format == ResultFormat.CSV:
            self._save_as_csv(results, output_path, **kwargs)
        elif format == ResultFormat.JSONL:
            self._save_as_jsonl(results, output_path, **kwargs)
        elif format == ResultFormat.EXCEL:
            self._save_as_excel(results, output_path, **kwargs)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def load_results(self, 
                    input_path: Union[str, Path],
                    format: Optional[ResultFormat] = None) -> List[ProcessingResult]:
        """
        加载处理结果
        
        Args:
            input_path: 输入路径
            format: 输入格式，如果为None则根据文件扩展名推断
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        input_path = Path(input_path)
        
        if format is None:
            format = self._infer_format(input_path)
        
        if format == ResultFormat.JSON:
            return self._load_from_json(input_path)
        elif format == ResultFormat.JSONL:
            return self._load_from_jsonl(input_path)
        else:
            raise ValueError(f"不支持从格式 {format} 加载")
    
    def _save_as_json(self, results: List[ProcessingResult], output_path: Path, **kwargs):
        """保存为JSON格式"""
        indent = kwargs.get('indent', 4)
        ensure_ascii = kwargs.get('ensure_ascii', False)
        
        # 转换为字典列表
        data = [self._result_to_dict(result) for result in results]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=self.encoder, indent=indent, ensure_ascii=ensure_ascii)
    
    def _save_as_jsonl(self, results: List[ProcessingResult], output_path: Path, **kwargs):
        """保存为JSON Lines格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                data = self._result_to_dict(result)
                f.write(json.dumps(data, cls=self.encoder, ensure_ascii=False) + '\n')
    
    def _save_as_csv(self, results: List[ProcessingResult], output_path: Path, **kwargs):
        """保存为CSV格式"""
        if not results:
            return
        
        # 定义CSV字段
        fieldnames = [
            'article_id', 'title', 'file_path', 'status',
            'concepts_count', 'entities_count', 'keywords_count',
            'quality_score', 'importance_score', 'trending_score',
            'tags', 'categories', 'processing_time', 'processed_at',
            'errors'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = self._result_to_csv_row(result)
                writer.writerow(row)
    
    def _save_as_excel(self, results: List[ProcessingResult], output_path: Path, **kwargs):
        """保存为Excel格式"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("保存Excel格式需要安装pandas: pip install pandas openpyxl")
        
        # 转换为DataFrame
        data = []
        for result in results:
            row = self._result_to_csv_row(result)
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False, engine='openpyxl')
    
    def _load_from_json(self, input_path: Path) -> List[ProcessingResult]:
        """从JSON文件加载"""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [self._dict_to_result(item) for item in data]
    
    def _load_from_jsonl(self, input_path: Path) -> List[ProcessingResult]:
        """从JSON Lines文件加载"""
        results = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(self._dict_to_result(data))
        return results
    
    def _result_to_dict(self, result: ProcessingResult) -> Dict[str, Any]:
        """将ProcessingResult转换为字典"""
        data = result.dict()
        
        # 处理特殊字段
        if data.get('processed_at'):
            data['processed_at'] = data['processed_at'].isoformat() if isinstance(data['processed_at'], datetime) else data['processed_at']
        
        return data
    
    def _result_to_csv_row(self, result: ProcessingResult) -> Dict[str, Any]:
        """将ProcessingResult转换为CSV行"""
        row = {
            'article_id': result.article_id,
            'title': result.title,
            'file_path': result.file_path,
            'status': result.status.value,
            'quality_score': result.quality_score,
            'importance_score': result.importance_score,
            'trending_score': result.trending_score,
            'tags': '|'.join(result.tags) if result.tags else '',
            'categories': '|'.join(result.categories) if result.categories else '',
            'processing_time': result.processing_time,
            'processed_at': result.processed_at.isoformat() if result.processed_at else '',
            'errors': '|'.join(result.errors) if result.errors else ''
        }
        
        # 处理概念提取结果
        if result.concepts:
            row['concepts_count'] = len(result.concepts.concepts)
            row['entities_count'] = len(result.concepts.entities)
            row['keywords_count'] = len(result.concepts.keywords)
        else:
            row['concepts_count'] = 0
            row['entities_count'] = 0
            row['keywords_count'] = 0
        
        return row
    
    def _dict_to_result(self, data: Dict[str, Any]) -> ProcessingResult:
        """将字典转换为ProcessingResult"""
        # 处理日期时间字段
        if 'processed_at' in data and isinstance(data['processed_at'], str):
            try:
                data['processed_at'] = datetime.fromisoformat(data['processed_at'])
            except ValueError:
                data['processed_at'] = None
        
        return ProcessingResult(**data)
    
    def _infer_format(self, file_path: Path) -> ResultFormat:
        """根据文件扩展名推断格式"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            return ResultFormat.JSON
        elif suffix == '.jsonl':
            return ResultFormat.JSONL
        elif suffix == '.csv':
            return ResultFormat.CSV
        elif suffix in ['.xlsx', '.xls']:
            return ResultFormat.EXCEL
        else:
            raise ValueError(f"无法推断文件格式: {suffix}")
    
    def generate_summary_report(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """生成处理结果摘要报告"""
        if not results:
            return {"error": "没有处理结果"}
        
        total = len(results)
        successful = sum(1 for r in results if r.status.value == "completed")
        failed = sum(1 for r in results if r.status.value == "failed")
        
        # 计算平均分数
        quality_scores = [r.quality_score for r in results if r.quality_score is not None]
        importance_scores = [r.importance_score for r in results if r.importance_score is not None]
        
        # 统计概念数量
        total_concepts = sum(len(r.concepts.concepts) if r.concepts else 0 for r in results)
        total_entities = sum(len(r.concepts.entities) if r.concepts else 0 for r in results)
        total_keywords = sum(len(r.concepts.keywords) if r.concepts else 0 for r in results)
        
        # 统计分类
        all_categories = []
        for r in results:
            all_categories.extend(r.categories)
        
        category_counts = {}
        for cat in all_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return {
            "总体统计": {
                "总文章数": total,
                "成功处理": successful,
                "处理失败": failed,
                "成功率": f"{successful/total*100:.1f}%" if total > 0 else "0%"
            },
            "质量指标": {
                "平均质量分": f"{sum(quality_scores)/len(quality_scores):.3f}" if quality_scores else "N/A",
                "平均重要性分": f"{sum(importance_scores)/len(importance_scores):.3f}" if importance_scores else "N/A"
            },
            "内容统计": {
                "总概念数": total_concepts,
                "总实体数": total_entities,
                "总关键词数": total_keywords,
                "平均概念数": f"{total_concepts/successful:.1f}" if successful > 0 else "N/A"
            },
            "分类统计": category_counts,
            "处理时间": {
                "总处理时间": f"{sum(r.processing_time for r in results if r.processing_time):.2f}秒",
                "平均处理时间": f"{sum(r.processing_time for r in results if r.processing_time)/total:.2f}秒" if total > 0 else "N/A"
            }
        }
    
    def save_summary_report(self, results: List[ProcessingResult], output_path: Union[str, Path]):
        """保存摘要报告"""
        report = self.generate_summary_report(results)
        
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)