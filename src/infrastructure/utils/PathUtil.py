import os
import sys
from pathlib import Path
from typing import Union


# 获取当前项目的目录路径
def get_project_base_dir():
    """
    获取项目根目录的绝对路径。
    无论哪个模块调用，都能正确定位根目录。
    """
    # 1. 获取当前工具函数所在文件（PathUtil.py）的绝对路径
    current_file = Path(__file__).resolve()

    # 2. 向上推导根目录：从 PathUtil.py 到 src/，再到根目录

    project_root = current_file.parent.parent.parent.parent

    # 3. 验证根目录（可选，增加容错性）
    if not (project_root / "src").exists():  # 检查根目录下是否有 src 文件夹（确保正确）
        raise FileNotFoundError(f"无法定位项目根目录，请检查路径推导逻辑。当前推导：{project_root}")
    return project_root

def concat_path(base_path: Union[str, Path], *paths: str) -> Path:
    """
    拼接基路径和后续需要拼接的路径(支持多个,而且最后一个参数如果不以/结束表示是文件名,如果以/结束表示还是目录),输入的路径全部使用linux风格,全部由/分隔
    返回的是跨平台的路径,对于windows会转换成\
    :param base_path: 基础路径
    :param paths: 其他路径部分
    :return: 拼接后的完整路径
    """
    if type(base_path) is not str:
        base_path = str(base_path)
    # 处理基础路径，去除开头和结尾的斜杠
    if base_path.startswith('/'):
        base_path = base_path[1:]
    if base_path.endswith('/'):
        base_path = base_path[:-1]

    # 存储所有路径组件
    path_components = [base_path] if base_path else []

    # 处理其他路径部分
    for path in paths:
        if not path:  # 跳过空字符串
            continue

        # 去除开头的斜杠
        if path.startswith('/'):
            path = path[1:]

        # 如果路径以斜杠结尾，说明是目录，去掉结尾斜杠
        if path.endswith('/'):
            path = path[:-1]

        # 分割路径并添加到组件列表
        if path:  # 确保不是空字符串
            path_components.extend(path.split('/'))

    # 使用 os.path.join 进行跨平台路径拼接
    if not path_components:
        return ''

    result = os.path.join(*path_components)

    # 检查最后一个参数是否以/结尾，如果是则添加目录分隔符
    if paths and paths[-1].endswith('/'):
        result += os.sep

    return Path(result)


# 测试用例
if __name__ == "__main__":
    # 测试基本功能
    print(concat_path(get_project_base_dir(),"/test/test2"))
    print(concat_path("home", "user", "documents"))  # home/user/documents (Linux) 或 home\user\documents (Windows)
    print(concat_path("/home", "user/", "documents"))  # home/user/documents
    print(concat_path("home/", "/user", "documents/"))  # home/user/documents/ (结尾有分隔符，表示目录)
    print(concat_path("", "user", "file.txt"))  # user/file.txt
    print(concat_path("home", "user", "file.txt"))  # home/user/file.txt
    print(concat_path("/home/user/", "/documents/", "file.txt"))  # home/user/documents/file.txt
