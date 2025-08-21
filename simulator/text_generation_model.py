import math
import numpy as np
import random
import sys
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_client import ExampleLLM
from model_evaluator import evaluate_model

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
        self.price = config.price
        self.model_keys = config.model_keys
        self.model_costs = config.model_costs
        self.llms = [ExampleLLM(key) for key in self.model_keys]

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

    def get_price(self, t: int, mechanism_info: Optional[Dict] = None) -> float:
        """
        获取当前时间步的真实价格, 根据真实token使用乘以单个price计算
        
        Args:
            t: 当前时间步
            mechanism_info: 机制信息，包含历史真实使用模型列表等
            
        Returns:
            float: 当前的报价
        """
        # 根据时间步t获取对应的token使用情况和模型索引（t从1开始，需要转换为列表索引）
        if t < len(self.prompt_tokens_by_time) and t < len(self.completion_tokens_by_time) and t < len(self.model_idx_by_time):
            prompt_tokens = self.prompt_tokens_by_time[t]
            completion_tokens = self.completion_tokens_by_time[t]
            model_idx = self.model_idx_by_time[t]
            model_key = self.model_keys[model_idx]
            
            # 获取真实价格
            pricing = MODEL_PRICING.get(model_key, {"input": self.price, "output": self.price})
            
            # 计算真实价格
            real_price = prompt_tokens * pricing["input"] + completion_tokens * pricing["output"]
            self.price = real_price
            return real_price
        else:
            # 如果时间步t超出范围，返回默认价格
            return self.price

    def get_normal_model_key(self) -> str:
        """
        获取服务商正常被要求使用的模型
        """
        normal_models = {
            1: "o3-mini",
            2: "gpt-4o-mini", 
            3: "gpt-4o"
        }
        return normal_models.get(self.provider_id, self.model_keys[0])


    def _get_cheapest_model_idx(self) -> int:
        """获取最便宜模型的索引"""
        min_cost_idx = 0
        min_cost = float('inf')
        
        for i, model_key in enumerate(self.model_keys):
            pricing = MODEL_PRICING.get(model_key, {"input": self.price, "output": self.price})
            # 使用平均价格作为比较标准
            avg_price = (pricing["input"] + pricing["output"]) / 2
            if avg_price < min_cost:
                min_cost = avg_price
                min_cost_idx = i
                
        return min_cost_idx
    
    def delegate_provider(self, phase: int, t: int, second_best_reward=None, R=None) -> Dict:
        """产生reward的函数，根据不同阶段采用不同策略
        
        Args:
            phase: 阶段编号 (1, 2, 或其他)
            t: 当前时间步
            second_best_reward: 第二好的reward值（阶段2使用）
            R: 阶段2的委托次数限制（从user中传入）
            
        Returns:
            Dict: {
                "reward": float,            # 评估分数
                "price": float,             # 使用该模型产生的价格（用户可见）
                "tokens": Tuple[int, int],  # (prompt_token, completion_token)
            }
        """
            
        # 根据阶段选择模型策略
        if phase == 1:
            # 阶段一：永远使用真实模型
            model_key = self.get_normal_model_key()
            model_idx = self.model_keys.index(model_key)
            
        elif phase == 2:
            # 阶段二：首先使用真实模型，当累积reward达到R*second_best_reward时使用最便宜模型
            if second_best_reward is None:
                second_best_reward = 0.0
            if R is None:
                R = 0 # 默认值，但应该从user中传入
                
            threshold = R * second_best_reward
            
            # 线程安全地读取cumulative_reward
            with self._lock:
                current_cumulative_reward = self.cumulative_reward
            
            if current_cumulative_reward < threshold:
                # 使用真实模型
                model_key = self.get_normal_model_key()
                model_idx = self.model_keys.index(model_key)
            else:
                # 使用最便宜的模型
                model_idx = self._get_cheapest_model_idx()
                model_key = self.model_keys[model_idx]
                
        else:
            # 其他阶段：使用最便宜的模型
            model_idx = self._get_cheapest_model_idx()
            model_key = self.model_keys[model_idx]
        
        # 调用evaluate_model函数进行评估
        from utils import evaluate_model
        reward, prompt_tokens, completion_tokens = evaluate_model(model_key)
        
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
        
        return {
            "reward": float(reward),
            "price": float(price),
            "tokens": (int(prompt_tokens), int(completion_tokens))
        }
    

    