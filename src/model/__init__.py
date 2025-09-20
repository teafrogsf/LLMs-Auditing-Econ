from decimal import Clamped
import openai
import os
from typing import Tuple, Any
from openai import OpenAI
import openai

import os
from dotenv import load_dotenv


load_dotenv()


MODEL_PRICING = {
    "gpt-4o": {"input": 2.5/1_000_000, "output": 10/1_000_000},
    "gpt-4": {"input": 30/1_000_000, "output": 60/1_000_000},
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.6/1_000_000},
    "o1-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "o3-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "gpt-35-turbo": {"input": 0.5/1_000_000, "output": 1.5/1_000_000},
    "qwen-max": {"input": 1.6/1_000_000, "output": 6.4/1_000_000},
    "deepseek-v3": {"input": 0.07/1_000_000, "output": 1.10/1_000_000},
    "deepseek-r1": {"input": 0.14/1_000_000, "output": 2.19/1_000_000},
    "o1": {"input": 15/1_000_000, "output": 60/1_000_000},
    "gpt-5": {"input": 1.25/1_000_000, "output": 10.0/1_000_000}
}

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
            base_url=os.getenv("QWEN_BASE_URL")),
            OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            ),
            openai.AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_GPT5_ENDPOINT"),
                api_key=os.getenv("AZURE_GPT5_KEY"),
                api_version="2024-12-01-preview"),
            ]

        self.key_client_map = {
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
                             'gpt-5': 4,
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
    llm = ExampleLLM("o1")
    
    # 测试调用
    response, prompt_tokens, completion_tokens = llm.call_llm("你好，请介绍一下你自己")
    print("回复:", response)
    print("输入token数:", prompt_tokens)
    print("输出token数:", completion_tokens)