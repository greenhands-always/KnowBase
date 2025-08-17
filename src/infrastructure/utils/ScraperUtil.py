import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
def get_markdown_from_url(url: str, content_selector: str) -> str:
    """
    从指定的URL获取Markdown格式的文本内容。

    此方法效率更高，因为它直接发起HTTP请求并解析HTML，无需自动化一个真实的浏览器。

    :param url: 目标URL。
    :param content_selector: 用于定位主要内容容器的CSS选择器。
    :return: Markdown格式的文本内容。
    """
    try:
        # 步骤 1: 发起HTTP请求，获取页面HTML内容
        headers = {
            # 模拟浏览器User-Agent，防止被一些网站拦截
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 如果请求失败 (状态码 4xx 或 5xx), 则抛出异常
        response.encoding = response.apparent_encoding # 自动识别并设置正确的编码

        # 步骤 2: 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # 步骤 3: 使用CSS选择器定位正文容器
        # CSS选择器通常比XPath更简洁易读。
        # 例如，如果原来的XPath是 //div[@id="content_views"]，对应的CSS选择器就是 '#content_views'。
        # 如果原来的XPath是 //div[@class="article-content"]，对应的CSS选择器就是 'div.article-content'。
        content_div = soup.select_one(content_selector)

        if not content_div:
            return "错误：无法通过提供的选择器找到内容容器。"

        # 步骤 4: 将定位到的HTML内容块直接转换为Markdown
        # markdownify库会自动处理各种HTML标签（如<h1>, <p>, <ul>, <a>, <img>等）
        # 并将它们转换为对应的Markdown语法。
        markdown_text = md(str(content_div), heading_style="ATX")

        return markdown_text.strip()

    except requests.exceptions.RequestException as e:
        return f"错误：请求URL时发生异常 - {e}"
    except Exception as e:
        return f"错误：处理过程中发生未知错误 - {e}"
