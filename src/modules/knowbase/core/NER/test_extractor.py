#!/usr/bin/env python3
"""
测试脚本：对比不同概念提取方法的效果
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_concept_extractor import EnhancedConceptExtractor
from llm_concept_extractor import LLMConceptExtractor


def create_test_files():
    """创建测试用的Markdown文件"""
    test_content = """
# 深度学习框架对比：PyTorch vs TensorFlow

## 引言

在人工智能和机器学习领域，深度学习框架扮演着至关重要的角色。本文将对比分析两个主流框架：PyTorch和TensorFlow。

## PyTorch简介

PyTorch是由Facebook AI Research开发的深度学习框架，具有以下特点：
- **动态计算图**：相比静态图框架更加灵活
- **Pythonic语法**：易于学习和使用
- **强大的GPU加速**：支持CUDA和cuDNN
- **丰富的预训练模型**：通过torchvision和transformers库

### 核心组件
- `torch.nn`：神经网络模块
- `torch.optim`：优化器（如Adam, SGD）
- `torch.utils.data`：数据加载工具
- `torchvision`：计算机视觉库

## TensorFlow简介

TensorFlow是Google开发的开源机器学习平台：
- **静态计算图**：优化性能更好
- **Keras API**：高级接口简化开发
- **TensorBoard**：可视化工具
- **TPU支持**：Google专用AI芯片

### TensorFlow 2.x新特性
- 即时执行（Eager Execution）
- Keras集成
- SavedModel格式
- TensorFlow Lite（移动端部署）

## 对比分析

| 特性 | PyTorch | TensorFlow |
|------|---------|------------|
| 学习曲线 | 较平缓 | 较陡峭 |
| 调试难度 | 简单 | 复杂 |
| 部署便利性 | 一般 | 优秀 |
| 社区支持 | 活跃 | 庞大 |

## 实际应用案例

### 计算机视觉
使用PyTorch实现ResNet50图像分类：
```python
import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)
model.eval()
```

### 自然语言处理
使用TensorFlow实现BERT文本分类：
```python
import tensorflow as tf
from transformers import TFBertModel

model = TFBertModel.from_pretrained('bert-base-uncased')
```

## 结论

两个框架各有优势，选择取决于具体需求：
- **研究/原型开发**：推荐使用PyTorch
- **生产环境部署**：推荐使用TensorFlow
"""

    # 创建临时目录和文件
    temp_dir = tempfile.mkdtemp()
    test_files = []
    
    for i in range(3):
        file_path = os.path.join(temp_dir, f"test_article_{i+1}.md")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        test_files.append(file_path)
    
    return temp_dir, test_files


def test_traditional_vs_llm():
    """测试传统方法vs大模型方法"""
    print("=== 概念提取方法对比测试 ===\n")
    
    # 创建测试文件
    temp_dir, test_files = create_test_files()
    print(f"创建测试文件：{len(test_files)}个")
    
    try:
        # 测试增强版提取器
        print("\n--- 大模型方法（增强版） ---")
        
        # 使用OpenAI（需要API密钥）
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            extractor = EnhancedConceptExtractor(
                provider="openai",
                model="gpt-4o-mini",
                api_key=api_key,
                rate_limit=2.0
            )
            
            concepts = extractor.process_directory(temp_dir, limit=1)
            
            print(f"提取概念数量: {len(concepts)}")
            print("前10个概念:")
            for concept, freq in sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {concept}: {freq}")
            
        else:
            print("未设置OPENAI_API_KEY，跳过OpenAI测试")
        
        # 测试Ollama（本地模型）
        print("\n--- 大模型方法（Ollama本地） ---")
        try:
            extractor = EnhancedConceptExtractor(
                provider="ollama",
                model="llama3.1",
                base_url="http://localhost:11434",
                rate_limit=0.5
            )
            
            concepts = extractor.process_directory(temp_dir, limit=1)
            
            print(f"提取概念数量: {len(concepts)}")
            print("前10个概念:")
            for concept, freq in sorted(concepts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {concept}: {freq}")
                
        except Exception as e:
            print(f"Ollama测试失败: {e}")
            print("请确保Ollama正在运行: ollama run llama3.1")
    
    finally:
        # 清理测试文件
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\n清理测试文件: {temp_dir}")


def test_single_file():
    """测试单个文件的概念提取"""
    print("\n=== 单文件测试 ===\n")
    
    # 创建测试内容
    test_content = """
# 人工智能技术综述

## 核心概念

机器学习（Machine Learning）是人工智能的一个分支，通过数据训练模型来实现智能决策。深度学习（Deep Learning）使用多层神经网络，特别是Transformer架构，实现了突破性进展。

### 关键技术
- **神经网络**：包括CNN、RNN、LSTM等架构
- **注意力机制**：Self-Attention和Multi-Head Attention
- **预训练模型**：BERT、GPT、T5等大型语言模型
- **强化学习**：Q-Learning、Policy Gradient等方法

## 应用场景

计算机视觉领域使用卷积神经网络（CNN）进行图像识别，自然语言处理使用BERT进行文本理解。生成对抗网络（GAN）在图像生成方面表现出色。
"""
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_content)
        temp_file = f.name
    
    try:
        # 测试文件存在
        if os.path.exists(temp_file):
            print(f"测试文件: {temp_file}")
            
            # 使用Ollama测试
            try:
                extractor = EnhancedConceptExtractor(
                    provider="ollama",
                    model="llama3.1"
                )
                
                concepts = extractor.extract_concepts_from_file(temp_file)
                
                print(f"提取概念数量: {len(concepts)}")
                print("提取的概念:")
                for concept in concepts:
                    print(f"  - {concept}")
                    
            except Exception as e:
                print(f"测试失败: {e}")
                
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def demo_usage():
    """演示基本用法"""
    print("\n=== 使用示例 ===\n")
    
    print("1. 基本使用:")
    print("   from enhanced_concept_extractor import EnhancedConceptExtractor")
    print("   extractor = EnhancedConceptExtractor(provider='openai')")
    print("   concepts = extractor.process_directory('./articles')")
    print("   extractor.save_results(concepts, 'my_analysis')")
    
    print("\n2. 命令行使用:")
    print("   python enhanced_concept_extractor.py --provider openai --limit 10")
    print("   python enhanced_concept_extractor.py --provider ollama --model llama3.1")
    
    print("\n3. 环境变量设置:")
    print("   export OPENAI_API_KEY='your-key-here'")
    print("   export ANTHROPIC_API_KEY='your-key-here'")
    print("   export GEMINI_API_KEY='your-key-here'")


if __name__ == "__main__":
    print("大模型概念提取器测试脚本")
    print("=" * 50)
    
    # 检查环境
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("OpenAI")
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("Anthropic")
    if os.getenv("GEMINI_API_KEY"):
        providers.append("Google Gemini")
    
    print(f"检测到的提供商: {', '.join(providers) if providers else '无'}")
    print("Ollama本地模型: 需要手动启动 (ollama run llama3.1)")
    
    # 运行测试
    test_single_file()
    test_traditional_vs_llm()
    demo_usage()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("如需实际使用，请设置相应的API密钥并运行enhanced_concept_extractor.py")