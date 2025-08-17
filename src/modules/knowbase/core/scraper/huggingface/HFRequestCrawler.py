import requests
from bs4 import BeautifulSoup
import markdownify
from urllib.parse import urljoin
import time
import re
from typing import List, Dict, Union
import json
import os
from pathlib import Path
from src.infrastructure.utils import PathUtil


class HuggingFaceBlogScraper:
    def __init__(self):
        self.base_url = "https://huggingface.co"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def scrape_blog_posts(self, sort_method: str = "trending", max_pages: int = None) -> List[Dict]:
        """
        爬取博客文章列表
        :param sort_method: trending/recent
        :param max_pages: 最大爬取页数，None表示爬取所有页
        :return: 文章信息列表
        """
        if sort_method not in ["trending", "recent"]:
            raise ValueError("sort_method must be either 'trending' or 'recent'")

        if sort_method == "recent" and max_pages is None:
            max_pages = 10  # 默认只爬取10页recent文章

        all_posts = []
        page = 1
        has_more = True

        while has_more:
            url = f"{self.base_url}/blog/community?sort={sort_method}&p={page}"
            print(f"正在爬取第 {page} 页: {url}")

            try:
                response = self.session.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

                # 解析文章列表
                posts = self._parse_blog_list(soup)
                if not posts:
                    has_more = False
                    break

                all_posts.extend(posts)

                # 检查是否还有更多页面
                if max_pages is not None and page >= max_pages:
                    break

                page += 1
                time.sleep(1)  # 礼貌性延迟

            except Exception as e:
                print(f"爬取第 {page} 页时出错: {str(e)}")
                break

        return all_posts

    def _parse_blog_list(self, soup: BeautifulSoup) -> List[Dict]:
        """
        解析博客列表页
        """
        posts = []

        # 查找所有文章元素
        article_links = soup.select('a[href^="/blog/"][class*="px-3 py-2"]')

        for link in article_links:
            try:
                # 提取文章URL
                relative_url = link['href']
                full_url = urljoin(self.base_url, relative_url)

                # 提取标题
                title = link.select_one('h4').get_text(strip=True)

                # 提取作者信息
                author_tag = link.select_one('object a[href^="/"]')
                author_name = author_tag.get_text(strip=True) if author_tag else "Unknown"
                author_url = urljoin(self.base_url, author_tag['href']) if author_tag else ""

                # 提取发布时间
                time_tag = link.select_one('time')
                publish_time = time_tag['datetime'] if time_tag else ""
                publish_time_text = time_tag.get_text(strip=True) if time_tag else ""

                # 提取点赞数
                likes_tag = link.select_one('svg[fill="transparent"] + span')
                likes = likes_tag.get_text(strip=True) if likes_tag else "0"
                likes = int(re.sub(r'\D', '', likes)) if likes else 0

                posts.append({
                    "title": title,
                    "url": full_url,
                    "author": author_name,
                    "author_url": author_url,
                    "publish_time": publish_time,
                    "publish_time_text": publish_time_text,
                    "likes": likes
                })

            except Exception as e:
                print(f"解析文章时出错: {str(e)}")
                continue

        return posts

    def scrape_blog_content(self, url: str) -> Dict:
        """
        爬取单篇博客的详细内容
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # 提取主要内容
            content_div = soup.select_one('div.prose') or soup.select_one('article') or soup.select_one('div.content')

            if not content_div:
                return {"error": "无法找到内容区域"}

            # 转换为Markdown
            markdown_content = markdownify.markdownify(
                str(content_div),
                heading_style="ATX",
                autolinks=True,
                bullets='-'
            )
            # 去除开头头像和点赞等无关内容
            pattern = r'\[Back to Articles\]\(/blog\).*?Follow\]\(/[^)]*\)'

            markdown_content = re.sub(pattern, '', markdown_content, flags=re.DOTALL)

            # 清理Markdown内容
            markdown_content = self._clean_markdown(markdown_content)

            return {
                "content": markdown_content,
                "content_length": len(markdown_content)
            }

        except Exception as e:
            return {"error": str(e)}

    # def _clean_markdown(self, text: str) -> str:
    #     """
    #     清理Markdown文本
    #     """
    #     # 移除连续空行
    #     text = re.sub(r'\n{3,}', '\n\n', text)
    #     # 修复代码块
    #     text = re.sub(r'```\s*\n\s*\n', '```\n', text)
    #     text = re.sub(r'\n\s*\n```', '\n```', text)
    #     # 移除HTML注释
    #     text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    #     # 移除特殊字符
    #     text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    #
    #     return text.strip()
    def _clean_markdown(self, text: str) -> str:
        # 移除多余的空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 规范化换行
        text = re.sub(r'\n\s+\n', '\n\n', text)
        # 移除代码块前后的多余换行
        text = re.sub(r'(\n{3,})(```)', r'\n\2', text)
        text = re.sub(r'(```)(\n{3,})', r'\1\n', text)
        return text.strip()

    def save_to_json(self, data: List[Dict], filename: str):
        """
        保存数据到JSON文件
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def save_posts(trending_posts, limit=10,path:Union[str, Path]=None):
    if trending_posts:
        if path is None:
            save_dir = PathUtil.concat_path(PathUtil.get_project_base_dir(),"result/Huggingface/Blog")
        else:
            if type(path) is str:
                save_dir = Path(path)
            else:
                save_dir = path
        # 创建保存目录
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"找到 {len(trending_posts)} 篇文章，开始批量爬取...")

        successful_count = 0
        failed_count = 0
        trending_posts = trending_posts[:limit]  # 限制为前10篇文章
        for i, post in enumerate(trending_posts, 1):
            print(f"\n正在处理第{i} 篇文章")
            print(f"文章标题: {post['title']}")
            print(f"文章链接: {post['url']}")

            try:
                content = scraper.scrape_blog_content(post['url'])

                if "content" in content:
                    # 创建安全的文件名（移除特殊字符）
                    safe_title = "".join(c for c in post['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_title = safe_title.replace(' ', '_')[:100]  # 限制文件名长度

                    # 如果标题为空，使用索引作为文件名
                    if not safe_title:
                        safe_title = f"blog_post_{i}"

                    filename = f"{safe_title}.md"
                    filepath = save_dir / filename

                    # 如果文件名重复，添加数字后缀
                    counter = 1
                    while filepath.exists():
                        filename = f"{safe_title}_{counter}.md"
                        filepath = save_dir / filename
                        counter += 1

                    # 保存文章内容
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"# {post['title']}\n\n")
                        f.write(f"**原文链接**: {post['url']}\n\n")
                        f.write("---\n\n")
                        f.write(content["content"])

                    print(f"✓ 文章保存成功: {filepath}")
                    print(f"  内容长度: {content['content_length']} 字符")
                    successful_count += 1

                else:
                    print(f"✗ 爬取文章内容失败: {content.get('error', '未知错误')}")
                    failed_count += 1

            except Exception as e:
                print(f"✗ 处理文章时发生错误: {str(e)}")
                failed_count += 1

            # 添加延时避免请求过于频繁
            import time
            time.sleep(1)

        print(f"\n批量爬取完成!")
        print(f"成功: {successful_count} 篇")
        print(f"失败: {failed_count} 篇")
        print(f"文章保存位置: {save_dir.absolute()}")

    else:
        print("没有找到待处理的文章列表")
if __name__ == '__main__':
    scraper = HuggingFaceBlogScraper()
    base_dir = PathUtil.get_project_base_dir()
    result_base_dir = PathUtil.concat_path(base_dir, "result/Huggingface")
    trending_json_file_path = PathUtil.concat_path(result_base_dir,"huggingface_trending_posts.json")
    recent_json_file_path = PathUtil.concat_path(result_base_dir,"huggingface_recent_posts.json")

    trending_posts = List[Dict]

    # 配置选项
    USE_CACHED_DATA = True  # 设置为 True 从文件读取，False 重新爬取
    if USE_CACHED_DATA and os.path.exists(trending_json_file_path):
        # 从文件中读取文章列表
        try:
            with open(trending_json_file_path, "r", encoding="utf-8") as f:
                trending_posts = json.load(f)
            print(f"从缓存文件中读取了 {len(trending_posts)} 篇热门文章")
            print(f"数据来源: {trending_json_file_path}")

        except Exception as e:
            print(f"读取缓存文件失败: {str(e)}")
            print("将重新爬取文章列表...")
            USE_CACHED_DATA = False

    if not USE_CACHED_DATA or not os.path.exists(trending_json_file_path):
        # 爬取热门文章（所有页）
        trending_posts = scraper.scrape_blog_posts(sort_method="trending")
        print(f"爬取了 {len(trending_posts)} 篇热门文章")
        scraper.save_to_json(trending_posts, trending_json_file_path)

        # 爬取最新文章（前10页）
        recent_posts = scraper.scrape_blog_posts(sort_method="recent", max_pages=10)
        print(f"爬取了 {len(recent_posts)} 篇最新文章")
        scraper.save_to_json(recent_posts, recent_json_file_path)
    if trending_posts:
        # save_posts(trending_posts)
        save_posts(trending_posts, len(trending_posts),PathUtil.concat_path(result_base_dir,"Blog/Trending"))