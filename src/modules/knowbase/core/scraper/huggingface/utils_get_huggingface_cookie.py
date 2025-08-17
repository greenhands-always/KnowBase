# -*- coding: utf-8 -*-
# @Time       : 2024/11/7 16:18
# @Author     : Marverlises
# @File       : get_huggingface_cookie.py
# @Description: PyCharm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from configReader import ConfigReader
from utils import init_driver
import time
import pickle  # 用于保存和加载 cookies

config = ConfigReader('config.ini')
USERNAME = config.get('huggingface','username')
PASSWORD = config.get('huggingface','password')

driver = init_driver(headless=True,chrome_exe_path=r"C:\software\chromex64\chrome.exe")
driver.get("https://huggingface.co/login")
time.sleep(2)

# 输入用户名和密码
username = driver.find_element(By.NAME, "username")
password = driver.find_element(By.NAME, "password")

# 将 Hugging Face 账号的用户名和密码输入到对应的输入框
username.send_keys(USERNAME)  # 替换为实际的用户名
password.send_keys(PASSWORD)  # 替换为实际的密码
password.send_keys(Keys.RETURN)

# 等待登录完成
time.sleep(5)  # 根据网络情况调整等待时间

# 获取 cookies 并保存到文件中
cookies = driver.get_cookies()
with open("huggingface_cookies.pkl", "wb") as file:
    pickle.dump(cookies, file)
print("Cookies 已保存为 huggingface_cookies.pkl")

# 关闭浏览器
driver.quit()


