# -*- coding: utf-8 -*-
# @Time       : 2024/11/7 10:15
# @Author     : Marverlises
# @File       : SeleniumUtils.py
# @Description: PyCharm
import re
import json
import time
import requests
import random
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 根据自己的代理地址和端口填写，也可以不设置注释掉
os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'



def init_driver(headless=True,chrome_exe_path = None):
    """
    初始化WebDriver
    :return:    WebDriver
    """
    # 初始化WebDriver
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--start-maximized')
    # options.add_argument('--proxy-server=http://your_proxy_address:port')  # 设置代理地址和端口
    if chrome_exe_path:
        options.binary_location = chrome_exe_path # 设置Chrome浏览器路径
    # 使用webdriver-manager自动下载并启动chrome驱动
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service = service, options=options)
    return driver


# 清洗文本
def clean_text(text):
    """
    简单清洗文本
    :param text:    待清洗文本
    :return:        将换行符替换为空格后的文本
    """
    text = text.strip()
    return " ".join(text.replace('\n', ' ').split())


def parse_string(input_string):
    """
    从字符串中提取下载次数、大小和日期
    :param input_string:   '@OpenGVLab提供1692下载356.51KB2024-07-27更新'
    :return:                {’downloads‘: 1692, 'size': 356.51KB, 'last_update': '2024-07-27'}
    """
    # 使用正则表达式匹配下载次数、大小和日期
    downloads_match = re.search(r'提供(\d+)下载', input_string)
    size_match = re.search(r'下载(.*?)B', input_string)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})更新', input_string)

    # 提取匹配结果，如果匹配不到则设置为 None
    downloads = int(downloads_match.group(1)) if downloads_match else None
    size = size_match.group(1) + 'B' if size_match else None
    last_update = date_match.group(1) if date_match else None

    # 将结果组装成字典
    result = {
        'downloads': downloads,
        'size': size,
        'last_update': last_update
    }
    return result


def read_json_file(file_path):
    """
    读取json文件
    :param file_path:   文件路径
    :return:            json数据
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        # 尝试用utf-8编码打开文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data


def save_json_data(data, save_path):
    """
    保存数据
    :param data:        数据
    :param save_path:   保存路径
    :return:            None
    """
    try:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise Exception(f"保存数据失败：{e}")


# 获取代理列表
def fetch_proxy_list():
    """
    获取代理列表
    :return:    代理列表
    """
    url = 'https://521proxy.com:8090/getProxyIp?num=5&return_type=txt&lb=1&sb=&flow=1&regions=us&protocol=http'
    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()
        proxies_list = response.text.strip().split('\r\n')
        print(f"Fetched proxies: {proxies_list}")  # 添加打印语句
        return proxies_list
    except requests.exceptions.HTTPError as err:
        print(f"HTTP请求错误: {err}")
    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")


def get_random_proxy():
    """
    随机选择一个代理
    :return:   代理
    """
    proxy_list = fetch_proxy_list()
    if not proxy_list:
        return None
    proxy = random.choice(proxy_list)
    proxy = {
        'http': f'http://{proxy}'
    }
    print(f"Using proxy: {proxy}")
    return proxy


def extract_arxiv_link(input_string):
    """
    从字符串中提取arxiv链接
    :param input_string: 数据集介绍中的所有内容
    :return:             arxiv的pdf链接——'https://arxiv.org/pdf/2406.08418'
    """
    matches = re.findall(r'\b\d{4}\.\d{5}\b', input_string)
    result_set = set(matches)
    arxiv_link = 'https://arxiv.org/pdf/' + result_set.pop() if result_set else ""
    return arxiv_link


def extract_pdf_link(introduction):
    """
    查看是否有相关pdf链接
    :param introduction: 数据集介绍中的所有内容
    :return:             相应的pdf链接——'http://pix3d.csail.mit.edu/papers/pix3d_cvpr.pdf'
    """
    # http[s] + :// + 任意字符 + .pdf
    matches = re.findall(r'http[s]?://.*?\.pdf', introduction)
    result_set = set(matches)
    pdf_link = result_set.pop() if result_set else ""
    return pdf_link
