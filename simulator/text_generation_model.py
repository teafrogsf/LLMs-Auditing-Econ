import math
from re import T
from unittest.loader import VALID_MODULE_NAME
import numpy as np
import json
import random
import sys
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_client import ExampleLLM

MODEL_PRICING = {
    "gpt-4o": {"input": 2.5/1_000_000, "output": 10/1_000_000},
    "gpt-4": {"input": 30/1_000_000, "output": 60/1_000_000},
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.6/1_000_000},
    "o1-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "o1": {"input": 15/1_000_000, "output": 60/1_000_000},
    "o3-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "gpt-35-turbo": {"input": 0.5/1_000_000, "output": 1.5/1_000_000},
    "qwen-max": {"input": 1.6/1_000_000, "output": 6.4/1_000_000},
    "deepseek-v3": {"input": 0.07/1_000_000, "output": 1.10/1_000_000},
    "deepseek-r1": {"input": 0.14/1_000_000, "output": 2.19/1_000_000},
}
STRATEGIES = ['ours', 'honest', 'random', 'worst']

@dataclass
class ProviderConfig:
    """服务商配置"""
    provider_id: int
    price: float  # p_i
    model_keys: List[str]  # 支持的模型列表
    model_costs: List[float]  # 各模型的真实cost
    strategy: str


class Evaluator:
    def __init__(self, models, param=7) -> None:
        # self.models = list(MODEL_PRICING.keys())
        self.data = {model: [json.loads(line) for line in open(f'test_result/{model}_test_result.jsonl')] for model in models}
        self.task_ids = json.load(open('task_ids_shuffled.json'))
        self.param = param

    def get_item(self, model, t):
        result = self.data[model][self.task_ids[t]]
        score = result.get('score', 0.0) * self.param
        prompt_tokens = result.get('input_tokens', 0)
        completion_tokens = result.get('output_tokens', 0)
    
        return score, prompt_tokens, completion_tokens


class Provider:
    """API服务商类（支持多模型和真实LLM调用）"""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.provider_id = config.provider_id
        self.price = config.price
        self.model_keys = config.model_keys
        self.model_costs = config.model_costs
        # self.llms = [ExampleLLM(key) for key in self.model_keys]
        self.evaluator = Evaluator(self.model_keys)
        self.strategy = config.strategy
        if self.strategy not in STRATEGIES:
            raise ValueError(f'Strategy {self.strategy} is not supported.')
        # 历史记录
        self.history_costs = []  # 历史成本
        self.total_delegations = 0  # 总委托次数
        self.model_usage = [0] * len(self.model_keys)  # 各模型被调用次数
        self.cumulative_reward = 0.0
        
        # 按时间步存储token使用情况
        self.prompt_tokens_by_time = []  # 每个时间步的input tokens
        self.completion_tokens_by_time = []  # 每个时间步的output tokens
        self.model_idx_by_time = []  # 每个时间步使用的模型索引
        
        # 线程安全锁
        self._lock = threading.Lock()



    def set_cost(self, t: int, mechanism_info: Optional[Dict] = None) -> float:
        """
        设置当前时间步的成本，根据当前时间步t和历史真实使用模型列表计算
        其中η：
        - provider1: η = 0.2
        - provider2: η = 0.6
        - provider3: η = 0.4
        
        Args:
            t: 当前时间步
            mechanism_info: 机制信息，包含历史真实使用模型列表等
            
        Returns:
            float: 当前花费的成本c
        """
        # 定义不同provider的η值
        eta_values = {
            1: 0.2,  # provider1
            2: 0.6,  # provider2
            3: 0.4   # provider3
        }
        
        # 获取当前provider的η值，默认为1.0
        eta = eta_values.get(self.provider_id, 1.0)
        
        # 根据时间步t获取对应的token使用情况和模型索引（t从1开始，需要转换为列表索引）
        if t < len(self.prompt_tokens_by_time) and t < len(self.completion_tokens_by_time) and t < len(self.model_idx_by_time):
            prompt_tokens = self.prompt_tokens_by_time[t]
            completion_tokens = self.completion_tokens_by_time[t]
            model_idx = self.model_idx_by_time[t]
            model_key = self.model_keys[model_idx]
            
            # 获取真实价格
            pricing = MODEL_PRICING.get(model_key, {"input": self.price, "output": self.price})
            
            # 计算真实成本（price × token）
            real_cost = prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
            
            # 应用η因子：cost = 真实price × token × η
            cost = real_cost * eta
        else:
            # 如果时间步t超出范围，使用默认成本
            cost = 0.0
            
        self.history_costs.append(cost)
        return cost

    def get_price(self, t: int) -> float:
        """
        获取当前时间步的价格，使用服务商最好模型的单价*当前时间步的tokens
        
        Args:
            t: 当前时间步
            
        Returns:
            float: 当前的报价
        """
        # 根据时间步t获取对应的token使用情况
        if t < len(self.prompt_tokens_by_time) and t < len(self.completion_tokens_by_time):
            prompt_tokens = self.prompt_tokens_by_time[t]
            completion_tokens = self.completion_tokens_by_time[t]
            
            # 获取服务商最好模型的价格
            best_model_idx = self._get_best_model_idx()
            best_model_key = self.model_keys[best_model_idx]
            best_model_pricing = MODEL_PRICING.get(best_model_key, {"input": self.price, "output": self.price})
            
            # 使用最好模型的单价计算价格
            price = prompt_tokens * best_model_pricing["input"] + completion_tokens * best_model_pricing["output"]
            return price
        else:
            # 如果时间步t超出范围，返回默认价格
            return self.price

    def _get_best_model_idx(self) -> int:
        """获取最贵模型的索引"""
        return max(
            range(len(self.model_keys)),
            key=lambda i: (MODEL_PRICING[self.model_keys[i]]["input"] + MODEL_PRICING[self.model_keys[i]]["output"]) / 2
        )

    def _get_cheapest_model_idx(self) -> int:
        """获取最便宜模型的索引"""
        return min(
            range(len(self.model_keys)),
            key=lambda i: (MODEL_PRICING[self.model_keys[i]]["input"] + MODEL_PRICING[self.model_keys[i]]["output"]) / 2
        )
    
    def get_total_cost(self) -> float:
        """获取该provider的总真实成本"""
        return sum(self.history_costs)
    
    def run(self, phase: int, t: int, second_best_utility=None, R=None) -> Dict:
        """产生reward的函数，根据不同阶段采用不同策略
        
        Args:
            phase: 阶段编号 (1, 2, 或其他)
            t: 当前时间步
            second_best_utility: 第二好的reward值（阶段2使用）
            R: 阶段2的委托次数限制（从user中传入）
            
        Returns:
            Dict: {
                "reward": float,            # 评估分数
                "price": float,             # 使用该模型产生的价格（用户可见）
                "tokens": Tuple[int, int],  # (prompt_token, completion_token)
            }
        """
            
        if self.strategy == 'ours':
            if phase == 1:
                # 阶段一：永远使用最贵的
                model_idx = self._get_best_model_idx()
                model_key = self.model_keys[model_idx]   

            elif phase == 2:
                # 阶段二：首先使用真实模型，当累积reward达到R*second_best_utility时使用最便宜模型
                if second_best_utility is None:
                    second_best_utility = 0.0
                if R is None:
                    R = 0 # 默认值，但应该从user中传入
                    
                threshold = R * second_best_utility
                
                # 线程安全地读取cumulative_reward
                with self._lock:
                    current_cumulative_reward = self.cumulative_reward
                
                if current_cumulative_reward < threshold:
                    # 使用真实模型（也就是最好的模型）
                    model_idx = self._get_best_model_idx()
                    model_key = self.model_keys[model_idx]
                else:
                    # 使用最便宜的模型
                    model_idx = self._get_cheapest_model_idx()
                    model_key = self.model_keys[model_idx]
                    
            else:
                # 其他阶段：使用最便宜的模型
                model_idx = self._get_cheapest_model_idx()
                model_key = self.model_keys[model_idx]
        
        elif self.strategy == 'worst':
            model_idx = self._get_cheapest_model_idx()
            model_key = self.model_keys[model_idx]
        
        elif self.strategy == 'honest':
            model_idx = self._get_best_model_idx()
            model_key = self.model_keys[model_idx]

        elif self.strategy == 'random':
            model_idx = random.choice(range(len(self.model_keys)))
            model_key = self.model_keys[model_idx]
        
        else:
            raise ValueError(f'No strategy {self.strategy}')

        # 调用evaluate_model函数进行评估
        reward, prompt_tokens, completion_tokens = self.evaluator.get_item(model_key, t)
        
        # 使用线程锁保护共享数据的访问
        with self._lock:
            # 更新累积reward
            self.cumulative_reward += reward
            
            # 按时间步存储token使用情况和模型索引
            # 确保列表足够长以容纳当前时间步
            while len(self.prompt_tokens_by_time) <= t:
                self.prompt_tokens_by_time.append(0)
            while len(self.completion_tokens_by_time) <= t:
                self.completion_tokens_by_time.append(0)
            while len(self.model_idx_by_time) <= t:
                self.model_idx_by_time.append(0)
                
            # 按时间步索引存储数据
            self.prompt_tokens_by_time[t] = prompt_tokens
            self.completion_tokens_by_time[t] = completion_tokens
            self.model_idx_by_time[t] = model_idx
        
        # 计算价格，使用当前时间步
        price = self.get_price(t)
        # 记录当前成本
        self.set_cost(t)

        
        return {
            "reward": float(reward),
            "price": float(price),
            "tokens": (int(prompt_tokens), int(completion_tokens))
        }
    