import os
import csv
import spacy
from collections import defaultdict


def extract_concepts_from_markdown(file_path):
    """从Markdown文件中提取概念"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 去除Markdown标记（简单处理）
        content = content.replace('#', '').replace('*', '').replace('`', '')

        nlp_en = spacy.load("en_core_web_sm")
        keyword_list = nlp_en(content).ents


        return keyword_list
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return []


def process_directory(directory,limit =30):
    """处理目录下的所有Markdown文件"""
    concept_freq = defaultdict(int)
    cnt = 0
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                print(f"正在处理文件: {file_path}")
                cnt += 1
                if cnt > limit:
                    print("已达到处理文件数量限制，停止处理。")
                    return concept_freq
                concepts = extract_concepts_from_markdown(file_path)

                # 统计词频
                for concept in concepts:
                    concept_freq[concept] += 1

    return concept_freq


def save_concept_frequency(concept_freq, output_file):
    """将概念频率保存到CSV文件"""
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

def save_concept_frequency2(concept_freq, output_file):
    """将概念频率保存到CSV文件"""
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

    concept_freq = defaultdict(int)
    for concept, freq in sorted_concepts:
        concept_freq[concept] += freq

    # 按频率排序（从高到低）
    merged_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)

    # 写入文件
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["概念", "频率"])
        for concept, freq in merged_concepts:
            writer.writerow([str(concept), freq])
def main():
    # 设置目录路径
    directory = r'D:\code\ai-trend-summary\result\Huggingface\Blog\Trending'

    # 输出文件路径
    output_file = r'D:\code\ai-trend-summary\result\Huggingface\Blog\concept_frequency2.csv'

    # 处理目录并统计概念频率
    concept_freq = process_directory(directory)

    # 保存结果
    save_concept_frequency2(concept_freq, output_file)


if __name__ == '__main__':
    main()