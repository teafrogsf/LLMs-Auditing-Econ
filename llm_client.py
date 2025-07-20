import os
from typing import Tuple, Any
from openai import OpenAI
import openai


class SingletonClient:
    _instance = None

    def __init__(self):
        from dotenv import find_dotenv, load_dotenv
        dotenv_path = find_dotenv()
        loaded = load_dotenv(dotenv_path, override=True)
        
        self.clients = [openai.AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_FRANCECENTRAL_ENDPOINT"),
            api_key=os.getenv("AZURE_FRANCECENTRAL_KEY"),
            api_version="2024-12-01-preview"),
            openai.AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_EASTUS_ENDPOINT"),
            api_key=os.getenv("AZURE_EASTUS_KEY"),
            api_version="2024-12-01-preview"),
            OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
            OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.lkeap.cloud.tencent.com/v1",
            # base_url="https://api.deepseek.com",
            ),
            ]

        self.key_client_map = {"gpt-4o-mini": 0,
                             "gpt-4o": 0,
                             "gpt-4": 0,
                             "gpt-4o-mini": 0,
                             "gpt-35-turbo": 0,
                             "o1-mini": 1,
                             "o1": 1,
                             "o3-mini": 1,
                             'qwen-max': 2,
                             'deepseek-r1': 3,
                             'deepseek-v3': 3,
                             'deepseek-chat': 3,
                             'deepseek-reasoner': 3,
                             }

    @classmethod
    def get(cls):
        if SingletonClient._instance is None:
            SingletonClient._instance = SingletonClient()
        return SingletonClient._instance


class ExampleLLM:
    def __init__(self, selected_model_key):
        """
        Initializes the ExampleLLM class by creating an instance of the OpenAI client.
        """
        self.client = SingletonClient.get()
        self.temperature = 0.7
        self.top_p = 1
        self.selected_model_key = selected_model_key

    def call_llm(self, user_prompt: str) -> Tuple[str, int, int]:
        """
        Sends the user prompt to OpenAI's GPT-4o model and retrieves the solution.

        Args:
            user_prompt (str): The complete prompt including the initial context and problem statement.

        Returns:
            Tuple[str, int, int]: The LLM's response, prompt token count, and completion token count.
        """
        client = self.client.clients[self.client.key_client_map[self.selected_model_key]]
        completion = client.chat.completions.create(
            model=self.selected_model_key,
            messages=[{"role": "user", "content": user_prompt}],
            # temperature=self.temperature,
            # top_p=self.top_p
        )
        content = completion.choices[0].message.content
        usage = getattr(completion, 'usage', None)
        prompt_tokens = 0
        completion_tokens = 0
        if usage is not None:
            # 兼容pydantic对象和dict
            if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
            elif isinstance(usage, dict):
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
            elif hasattr(usage, 'to_dict'):
                usage_dict = usage.to_dict()
                prompt_tokens = usage_dict.get('prompt_tokens', 0)
                completion_tokens = usage_dict.get('completion_tokens', 0)
        return content, prompt_tokens, completion_tokens


# 使用示例
if __name__ == "__main__":
    # 创建LLM客户端实例
    llm = ExampleLLM("deepseek-v3")
    
    # 测试调用
    response, prompt_tokens, completion_tokens = llm.call_llm("你好，请介绍一下你自己")
    print("回复:", response)
    print("输入token数:", prompt_tokens)
    print("输出token数:", completion_tokens) 