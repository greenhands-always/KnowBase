from pydantic_ai import Agent
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
import os # 添加os模块用于加载环境变量
load_dotenv()
class DeepSeekOfficialProvider(BaseModel):
    api_key = "DEEPSEEK_API_KEY"  # 替换为实际 API 密钥
    base_url = "https://api.deepseek.com/v1"  # 国内常用 DeepSeek 服务地址


# 运行示例
if __name__ == "__main__":
    model = OpenAIModel(
        "deepseek-r1",
        provider='deepseek',
        )
    agent = Agent(
        model=model,
        )
    response = agent.run("你好")
    print(response)
