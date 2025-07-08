import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from llm_client import ExampleLLM
from twenty_four_game import generate_hard_24_problem, check_24_answer

MODEL_PRICING = {
    "gpt-4o": {"input": 2.5/1_000_000, "output": 10/1_000_000},
    "o1-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "o3-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "gpt-35-turbo-0125-60ktpm": {"input": 0.0005/1_000, "output": 0.0015/1_000},
    "qwen-max": {"input": 1.6/1_000_000, "output": 6.4/1_000_000},
    "deepseek-chat": {"input": 0.07/1_000_000, "output": 1.10/1_000_000},
    "deepseek-r1": {"input": 0.14/1_000_000, "output": 2.19/1_000_000},
}

@dataclass
class ProviderConfig:
    """服务商配置"""
    provider_id: int
    price: float  # p_i
    model_keys: List[str]  # 支持的模型列表
    model_costs: List[float]  # 各模型的真实cost

class Provider:
    """API服务商类（支持多模型和真实LLM调用）"""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.provider_id = config.provider_id
        self.model_keys = config.model_keys
        self.model_costs = config.model_costs
        self.llms = [ExampleLLM(key) for key in self.model_keys]

        # 历史记录
        self.history_costs = []  # 历史成本
        self.history_rewards = []  # 历史回报
        self.total_delegations = 0  # 总委托次数
        self.model_usage = [0] * len(self.model_keys)  # 各模型被调用次数
        self.token_history = []  # 记录每次调用的token数

        # 以输入40token、输出15token为基准估算price
        DEFAULT_INPUT_TOKENS = 40
        DEFAULT_OUTPUT_TOKENS = 15
        costs = []
        for key in self.model_keys:
            pricing = MODEL_PRICING.get(key, {"input": 0.0, "output": 0.0})
            cost = DEFAULT_INPUT_TOKENS * pricing["input"] + DEFAULT_OUTPUT_TOKENS * pricing["output"]
            costs.append(cost)
        self.price = float(np.median(costs)) if costs else 0.0

    def set_cost(self, t: int, mechanism_info: Optional[Dict] = None) -> float:
        """
        设置当前时间步的成本，根据最近一次模型调用的token数和模型key自动计算
        """
        if hasattr(self, 'last_model_idx') and hasattr(self, 'last_tokens'):
            model_key = self.model_keys[self.last_model_idx]
            prompt_tokens, completion_tokens = self.last_tokens
            pricing = MODEL_PRICING.get(model_key, {"input": self.price, "output": self.price})
            cost = prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
        else:
            cost = self.price
        self.history_costs.append(cost)
        return cost

    def solve_24(self, nums) -> Optional[str]:
        """
        依次用模型尝试解24点，优先用高级模型，若低级模型能解则偷偷替换
        返回表达式或None
        """
        prompt = f"请用加减乘除和括号将{nums}这四个数拼成24，要求每个数只能用一次，返回一个合法表达式。只返回表达式本身。"
        for idx, llm in enumerate(self.llms):
            try:
                answer, prompt_tokens, completion_tokens = llm.call_llm(prompt)
                expr = answer.strip().split('\n')[0]
                if check_24_answer(nums, expr):
                    self.last_model_idx = idx
                    self.model_usage[idx] += 1
                    self.last_tokens = (prompt_tokens, completion_tokens)
                    self.last_cost = self.set_cost(0)
                    self.token_history.append(self.last_tokens)
                    return expr
            except Exception as e:
                continue
        self.last_model_idx = len(self.llms) - 1  # 最后一个模型失败
        self.last_tokens = (0, 0)
        self.last_cost = self.price
        self.token_history.append(self.last_tokens)
        return None

    def generate_reward(self, cost: float) -> float:
        """
        reward=1（答对）或0（答错）
        """
        nums = generate_hard_24_problem()
        expr = self.solve_24(nums)
        if expr is not None:
            reward = 1.0
        else:
            reward = 0.0
        self.history_rewards.append(reward)
        self.total_delegations += 1
        return reward

    def get_average_reward(self) -> float:
        """获取历史平均回报"""
        if not self.history_rewards:
            return 0.0
        return float(np.mean(self.history_rewards))

    def get_recent_average_reward(self, recent_count: int) -> float:
        """获取最近n次的平均回报"""
        if not self.history_rewards:
            return 0.0
        recent_rewards = self.history_rewards[-recent_count:]
        return float(np.mean(recent_rewards)) 