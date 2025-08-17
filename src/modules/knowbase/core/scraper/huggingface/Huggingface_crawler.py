# -*- coding: utf-8 -*-
# @Time       : 2024/11/20 15:25
# @Author     : Marverlises
# @File       : Huggingface_crawler.py
# @Description: 爬取Huggingface数据集信息

import os
import logging
import json
import time
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests
import tqdm
import pickle
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

from infrastructure.utils import ScraperUtil
from src.config.configReader import ConfigReader

from infrastructure.utils.SeleniumUtils import init_driver, clean_text
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from src.scraper.xpaths import XPaths


class HuggingfaceCrawler:
    """
    爬取Huggingface数据集信息，首先需要爬取数据集链接，然后再爬取数据集信息，如果已经爬取了数据集链接，则直接爬取数据集信息
    数据集链接文件：hugging_face_organization_datasets_links.json
    """

    def __init__(self, headless=True,
                 organization_links_file_path='organization_links/hugging_faceorganization_links.json',
                 sort_method='downloads', save_dir='result/huggingface',
                 organization_datasets_links_save_file='hugging_face_organization_datasets_links.json',
                 logging_cookie_file_path='./huggingface_cookies.pkl'):
        """
        初始化
        :param headless:                        是否启用无头模式
        :param organization_links_file_path:    机构链接文件路径
        :param sort_method:                     排序方法-[updated, created, alphabetical, likes, downloads, rowsMost, rowsLeast]
        :param save_dir:                        保存目录
        :param organization_datasets_links_save_file:  机构数据集链接保存文件
        :param logging_cookie_file_path:        登录cookie文件路径
        """
        self.config = ConfigReader('config.ini')
        chrome_exe_path = self.config.get('huggingface', 'chrome_exe_path')
        self.driver = init_driver(headless, chrome_exe_path=chrome_exe_path)
        self.organization_links_file_path = organization_links_file_path
        self.sort_method = sort_method
        self.save_dir = save_dir
        self.organization_datasets_links_save_file = organization_datasets_links_save_file
        self.logging_cookie_file_path = logging_cookie_file_path
        # 初始化logger，初始化相关元素的Xpath
        self._init_logger(log_level=logging.INFO)
        self._init_relevant_element_xpath()
        # 创建保存截图的文件夹
        if not os.path.exists(f'{self.save_dir}/hugging_face_dataset_info_screenshots'):
            os.makedirs('result/huggingface/hugging_face_dataset_info_screenshots')
        self.screen_shot_save_path = f'{self.save_dir}/hugging_face_dataset_info_screenshots'

    def _init_relevant_element_xpath(self) -> None:
        """
        初始化相关元素的Xpath
        :return:
        """
        # =========================== _crawl_dataset_links相关的 ================================
        # expand all按钮
        self.expand_all_button_xpath = '//*[@id="datasets"]/div/div[2]/div/button'
        # 每一个数据集的div
        self.dataset_item_xpath = '//*[@id="datasets"]/div/div/article'
        # =========================== _crawl_dataset_info相关的 ================================
        # 如果需要填写表单，则填写表单的元素
        self.need_finish_form_xpath = '/html/body/div/main/div[2]/section[1]/div[1]/div/form'
        # 如果需要填写表单，则填写表单的元素
        self.form_items_xpath = '/html/body/div/main/div[2]/section[1]/div[1]/div/form/label'
        # 如果获取数据集详情需要填写表单并点击按钮
        self.finish_form_button_xpath = '/html/body/div/main/div[2]/section[1]/div[1]/div/form/div/button'
        # tags信息
        self.tags_info_xpath = '/html/body/div/main/div[1]/header/div/div[1]/div'
        # 右侧面板信息
        self.data_info_div_xpath = '//div[@class="flex flex-col flex-wrap xl:flex-row"]'
        # 下载量
        self.download_count_xpath = '/html/body/div/main/div[2]/section[2]/dl/dd'
        # 社区活跃
        self.community_xpath = '/html/body/div/main/div[1]/header/div/div[2]/div/a[last()]'
        # 点赞数
        self.like_xpath = '/html/body/div/main/div[1]/header/div/h1/div[3]/button[2]'

    def is_logged_in(self, timeout=10) -> bool:
        """
        检查是否已登录Hugging Face
        :param timeout: 超时时间（秒）
        :return: 是否已登录
        """
        try:
            # 访问设置页面
            self.driver.get("https://huggingface.co/settings")

            # 等待页面加载

            # 检查是否存在"Profile Settings"文本
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Profile Settings')]"))
            )
            return True
        except TimeoutException:
            # 尝试其他登录状态检查方法
            try:
                # 检查用户头像
                avatar = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//img[@alt='User avatar']"))
                )
                if avatar.is_displayed():
                    return True
            except TimeoutException:
                pass

            try:
                # 检查用户名显示
                username_display = WebDriverWait(self.driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'username')]"))
                )
                if username_display.is_displayed():
                    return True
            except TimeoutException:
                pass

            return False
        except Exception as e:
            logging.error(f"登录状态检查失败: {str(e)}")
            return False

    def perform_login(self, username=None, password=None) -> bool:
        """
        执行登录流程
        :param username: 用户名
        :param password: 密码
        :return: 是否登录成功
        """
        logging.info("开始登录流程...")
        try:
            # 导航到登录页面
            self.driver.get("https://huggingface.co/login")
            time.sleep(2)

            # 获取凭据
            config = ConfigReader('config.ini')
            login_username = username or config.get('huggingface', 'username')
            login_password = password or config.get('huggingface', 'password')

            if not login_username or not login_password:
                logging.error("无法获取用户名或密码")
                return False

            # 输入凭据
            username_input = self.driver.find_element(By.NAME, "username")
            password_input = self.driver.find_element(By.NAME, "password")

            username_input.send_keys(login_username)
            password_input.send_keys(login_password)
            password_input.send_keys(Keys.RETURN)

            # 等待登录完成 - 使用显式等待
            try:
                WebDriverWait(self.driver, 15).until(
                    lambda driver: self.is_logged_in()
                )
            except TimeoutException:
                logging.warning("登录状态验证超时，尝试继续...")

            # 检查登录是否成功
            if self.is_logged_in():
                # 保存新的Cookie
                cookies = self.driver.get_cookies()
                with open(self.logging_cookie_file_path, "wb") as file:
                    pickle.dump(cookies, file)
                logging.info(f"登录成功，Cookie已保存至: {self.logging_cookie_file_path}")
                return True
            else:
                # 尝试检查登录失败提示
                try:
                    error_msg = self.driver.find_element(By.XPATH, "//div[contains(@class, 'error')]")
                    logging.error(f"登录失败: {error_msg.text}")
                except NoSuchElementException:
                    logging.error("登录失败：未知原因")

                # 保存截图以供调试
                screenshot_path = os.path.join(self.save_dir, "login_failure.png")
                self.driver.save_screenshot(screenshot_path)
                logging.info(f"登录失败截图已保存至: {screenshot_path}")

                return False
        except Exception as e:
            logging.error(f"登录过程中发生错误: {str(e)}")
            return False

    def login(self, USERNAME=None, PASSWORD=None) -> bool:
        """
        登录Hugging Face或加载已保存的Cookie
        如果Cookie文件存在则直接加载，否则执行登录流程并保存Cookie
        """
        # 如果已有有效的登录状态，直接返回
        if self.is_logged_in(timeout=3):
            logging.info("当前会话已登录")
            return True

        # 检查Cookie文件是否存在
        if os.path.exists(self.logging_cookie_file_path):
            try:
                logging.info("尝试加载Cookie...")
                # 先访问首页清除可能的登录状态
                self.driver.get("https://huggingface.co")
                time.sleep(1)

                # 清除现有Cookie
                self.driver.delete_all_cookies()

                # 加载保存的Cookie
                with open(self.logging_cookie_file_path, "rb") as file:
                    cookies = pickle.load(file)
                    for cookie in cookies:
                        # 确保域名为.huggingface.co以覆盖所有子域
                        if 'domain' in cookie and not cookie['domain'].startswith('.'):
                            cookie['domain'] = f".{cookie['domain']}"
                        try:
                            self.driver.add_cookie(cookie)
                        except Exception as e:
                            logging.warning(f"添加Cookie失败: {str(e)}")

                # 刷新页面使Cookie生效
                self.driver.refresh()
                time.sleep(2)

                # 检查登录状态
                if self.is_logged_in():
                    logging.info("Cookie加载成功，已登录Hugging Face")
                    return True
                else:
                    logging.warning("Cookie已失效，需要重新登录")
            except Exception as e:
                logging.error(f"加载Cookie失败: {str(e)}")

        # 执行登录流程
        logging.info("执行登录流程...")
        return self.perform_login(username=USERNAME, password=PASSWORD)
    def get_markdown_from_url(url: str, content_selector: str) -> str:
        content_selector = ".blog-content"
        ScraperUtil.get_markdown_from_url(url, content_selector)


    def get_top_blog_list(self) -> dict:
        """
        获取热门博客列表
        :return: 博客URL和标题的字典
        """
        self.driver.get("https://huggingface.co/blog/zh")
        time.sleep(3)

        try:
            # 等待页面加载完成
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.XPATH, XPaths.TOP_BLOG_LIST))
            )

            info_div = self.driver.find_element(By.XPATH, XPaths.TOP_BLOG_LIST)
            article_dict = {}
            articles = info_div.find_elements(By.XPATH, ".//article")

            for article in articles:
                try:
                    link = article.find_element(By.TAG_NAME, "a")
                    url = link.get_attribute("href")

                    # 尝试获取标题
                    try:
                        title_element = link.find_element(By.XPATH, ".//h4")
                        title = title_element.text
                    except NoSuchElementException:
                        title = "无标题"

                    article_dict[url] = title
                    logging.info(f"找到博客: {title} - {url}")
                except Exception as e:
                    logging.error(f"处理博客元素时出错: {str(e)}")

            return article_dict
        except TimeoutException:
            logging.error("等待博客列表超时")
            return {}
        except Exception as e:
            logging.error(f"获取博客列表时出错: {str(e)}")
            return {}

    def _init_logger(self, log_level: int = logging.INFO) -> None:
        """
        初始化日志
        :return:
        """
        log_dir = './logs/HF'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, 'HF_crawl_log.log')
        logging.basicConfig(level=log_level,
                            format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                            handlers=[
                                logging.FileHandler(log_file),  # 写入文件
                                logging.StreamHandler()  # 控制台输出
                            ])
        logging.info("Start crawling HF dataset info")

    def __del__(self):
        """析构函数，关闭浏览器"""
        try:
            self.driver.quit()
            logging.info("WebDriver已关闭")
        except Exception as e:
            logging.error(f"关闭WebDriver时出错: {str(e)}")


if __name__ == '__main__':
    huggingface_crawler = HuggingfaceCrawler(headless=False,
                                             organization_links_file_path='organization_links/hugging_faceorganization_links.json',
                                             sort_method='downloads',
                                             save_dir='result/huggingface')

    # 登录
    if huggingface_crawler.login():
        logging.info("登录成功")

        # 获取博客列表
        blog_list = huggingface_crawler.get_top_blog_list()
        logging.info(f"获取到 {len(blog_list)} 篇博客")

        # 打印博客列表
        for url, title in blog_list.items():
            logging.info(f"{title}: {url}")
    else:
        logging.error("登录失败，无法继续操作")