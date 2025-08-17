import os
import json
import time
from pathlib import Path
from src.infrastructure.utils.PathUtil import get_project_base_dir, concat_path
from src.scraper.huggingface.HFRequestCrawler import HuggingFaceBlogScraper

def is_likely_truncated(filename, title):
    """
    判断文件名是否可能被截断
    """
    # 移除.md扩展名
    name_without_ext = filename.replace('.md', '')
    
    # 如果文件名长度接近100字符，很可能被截断
    if len(name_without_ext) >= 95:
        return True
    
    # 如果标题比文件名长很多，可能被截断
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_')
    
    if len(safe_title) > len(name_without_ext) + 10:
        return True
    
    return False

def create_proper_filename(title):
    """
    创建完整的文件名（不截断）
    """
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_title = safe_title.replace(' ', '_')
    return f"{safe_title}.md"

def fix_truncated_files():
    """
    修复被截断的文件名
    """
    base_dir = get_project_base_dir()
    result_base_dir = concat_path(base_dir, "result/Huggingface")
    trending_json_file_path = concat_path(result_base_dir, "huggingface_trending_posts.json")
    blog_dir = concat_path(result_base_dir, "Blog/Trending")
    
    # 读取文章数据
    with open(trending_json_file_path, 'r', encoding='utf-8') as f:
        trending_posts = json.load(f)
    
    scraper = HuggingFaceBlogScraper()
    
    # 获取现有文件列表
    existing_files = set()
    if blog_dir.exists():
        existing_files = {f.name for f in blog_dir.iterdir() if f.is_file() and f.suffix == '.md'}
    
    print(f"找到 {len(existing_files)} 个现有文件")
    
    # 找出需要重新处理的文章
    files_to_reprocess = []
    
    for i, post in enumerate(trending_posts):
        title = post['title']
        proper_filename = create_proper_filename(title)
        
        # 检查是否有对应的截断文件
        for existing_file in existing_files:
            if is_likely_truncated(existing_file, title):
                # 检查这个截断文件是否对应当前文章
                existing_name_part = existing_file.replace('.md', '')
                proper_name_part = proper_filename.replace('.md', '')
                
                # 如果截断的文件名是完整文件名的前缀
                if proper_name_part.startswith(existing_name_part):
                    files_to_reprocess.append({
                        'post': post,
                        'old_filename': existing_file,
                        'new_filename': proper_filename,
                        'index': i
                    })
                    break
    
    print(f"发现 {len(files_to_reprocess)} 个需要重新处理的文件")
    
    if not files_to_reprocess:
        print("没有发现需要修复的截断文件")
        return
    
    # 确认是否继续
    print("\n将要重新处理的文件:")
    for item in files_to_reprocess[:10]:  # 只显示前10个
        print(f"  {item['old_filename']} -> {item['new_filename']}")
    
    if len(files_to_reprocess) > 10:
        print(f"  ... 还有 {len(files_to_reprocess) - 10} 个文件")
    
    confirm = input(f"\n确认要重新爬取这 {len(files_to_reprocess)} 个文件吗? (y/N): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return
    
    # 开始重新处理
    successful_count = 0
    failed_count = 0
    
    for item in files_to_reprocess:
        post = item['post']
        old_filename = item['old_filename']
        new_filename = item['new_filename']
        
        print(f"\n正在处理: {post['title']}")
        print(f"旧文件: {old_filename}")
        print(f"新文件: {new_filename}")
        
        try:
            # 删除旧文件
            old_filepath = blog_dir / old_filename
            if old_filepath.exists():
                old_filepath.unlink()
                print(f"✓ 已删除旧文件: {old_filename}")
            
            # 爬取新内容
            content = scraper.scrape_blog_content(post['url'])
            
            if "content" in content:
                new_filepath = blog_dir / new_filename
                
                # 如果新文件名已存在，添加数字后缀
                counter = 1
                while new_filepath.exists():
                    name_part = new_filename.replace('.md', '')
                    new_filename_with_counter = f"{name_part}_{counter}.md"
                    new_filepath = blog_dir / new_filename_with_counter
                    counter += 1
                
                # 保存新文件
                with open(new_filepath, "w", encoding="utf-8") as f:
                    f.write(f"# {post['title']}\n\n")
                    f.write(f"**原文链接**: {post['url']}\n\n")
                    f.write("---\n\n")
                    f.write(content["content"])
                
                print(f"✓ 新文件保存成功: {new_filepath.name}")
                print(f"  内容长度: {content['content_length']} 字符")
                successful_count += 1
                
            else:
                print(f"✗ 爬取内容失败: {content.get('error', '未知错误')}")
                failed_count += 1
        
        except Exception as e:
            print(f"✗ 处理文件时发生错误: {str(e)}")
            failed_count += 1
        
        # 添加延时避免请求过于频繁
        time.sleep(1)
    
    print(f"\n修复完成!")
    print(f"成功: {successful_count} 个文件")
    print(f"失败: {failed_count} 个文件")

if __name__ == '__main__':
    fix_truncated_files()