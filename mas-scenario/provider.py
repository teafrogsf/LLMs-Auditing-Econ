import numpy as np
import random
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_client import ExampleLLM
from twenty_four_game import generate_hard_24_problem, check_24_answer

# 添加phybench-test目录到路径以导入EED评分功能
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phybench-test'))
from phybench_evaluation import PHYBenchEvaluator

MODEL_PRICING = {
    "gpt-4o": {"input": 2.5/1_000_000, "output": 10/1_000_000},
    "gpt-4": {"input": 30/1_000_000, "output": 60/1_000_000},
    "o1-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "o3-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "gpt-35-turbo-0125-60ktpm": {"input": 0.5/1_000_000, "output": 1.5/1_000_000},
    "qwen-max": {"input": 1.6/1_000_000, "output": 6.4/1_000_000},
    "deepseek-chat": {"input": 0.07/1_000_000, "output": 1.10/1_000_000},
    "deepseek-r1": {"input": 0.14/1_000_000, "output": 2.19/1_000_000},
}

@dataclass
class ProviderConfig:
    """服务商配置"""
    provider_id: int
    price: float  # p_i
    mu: float     # μ_i
    model_keys: List[str]  # 支持的模型列表
    model_costs: List[float]  # 各模型的真实cost

class Provider:
    """API服务商类（支持多模型和真实LLM调用）"""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.provider_id = config.provider_id
        self.mu = config.mu
        self.model_keys = config.model_keys
        self.model_costs = config.model_costs
        self.llms = [ExampleLLM(key) for key in self.model_keys]

        # 历史记录
        self.history_costs = []  # 历史成本
        self.history_rewards = []  # 历史回报
        self.total_delegations = 0  # 总委托次数
        self.model_usage = [0] * len(self.model_keys)  # 各模型被调用次数
        self.token_history = []  # 记录每次调用的token数
        
        # 初始化EED评估器
        self.eed_evaluator = PHYBenchEvaluator(models_to_test=self.model_keys)
        self.eed_dataset = None  # 延迟加载数据集

        # 以输入500token、输出750token为基准估算price
        DEFAULT_INPUT_TOKENS = 500
        DEFAULT_OUTPUT_TOKENS = 750
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


    def get_normal_model_key(self) -> str:
        """
        获取服务商正常被要求使用的模型
        服务商1: gpt-4o, 服务商2: o1-mini, 服务商3: deepseek-v3
        """
        normal_models = {
            1: "gpt-4o",
            2: "o1-mini", 
            3: "deepseek-v3"
        }
        return normal_models.get(self.provider_id, self.model_keys[0])
    
    def evaluate_with_specific_model(self, model_key: str) -> Tuple[float, int, int]:
        """
        使用指定模型评估随机样本
        
        Args:
            model_key: 模型键值
            
        Returns:
            Tuple[float, int, int]: (EED分数, 输入token数, 输出token数)
        """
        try:
            # 延迟加载数据集
            if self.eed_dataset is None:
                self.eed_dataset = self.eed_evaluator.load_dataset(None, None)
                if not self.eed_dataset:
                    print(f"Provider {self.provider_id}: 数据集加载失败")
                    return 0.0, 0, 0
            
            # 使用test_random_sample方法随机选择样本
            eed_score, prompt_tokens, completion_tokens = self.eed_evaluator.test_random_sample(model_key, self.eed_dataset)
            
            # 记录使用的模型和token信息
            if model_key in self.model_keys:
                self.last_model_idx = self.model_keys.index(model_key)
                self.model_usage[self.last_model_idx] += 1
            else:
                self.last_model_idx = 0
                
            self.last_tokens = (prompt_tokens, completion_tokens)
            self.token_history.append(self.last_tokens)
            
            return eed_score, prompt_tokens, completion_tokens
                
        except Exception as e:
            print(f"Provider {self.provider_id}: 使用模型 {model_key} 评估失败: {e}")
            return 0.0, 0, 0
    
    def generate_reward_with_tokens(self, honest_mode: bool = True, second_best_score: float = 0.0) -> Tuple[float, int, int]:
        """
        使用EED评分机制生成回报，并返回token信息
        
        Args:
            honest_mode: 是否诚实模式（第一轮为True，第二轮为False）
            second_best_score: 次优服务商的EED分数（第二轮使用）
            
        Returns:
            Tuple[float, int, int]: (EED分数, 输入token数, 输出token数)
        """
        try:
            # 延迟加载数据集
            if self.eed_dataset is None:
                self.eed_dataset = self.eed_evaluator.load_dataset(None, None)
                if not self.eed_dataset:
                    print(f"Provider {self.provider_id}: 数据集加载失败")
                    return 0.0, 0, 0
            
            if honest_mode:
                # 第一轮：诚实模式，使用正常要求的模型
                normal_model = self.get_normal_model_key()
                if normal_model in self.model_keys:
                    eed_score, prompt_tokens, completion_tokens = self.evaluate_with_specific_model(normal_model)
                    self.history_rewards.append(eed_score)
                    self.total_delegations += 1
                    return eed_score, prompt_tokens, completion_tokens
                else:
                    # 如果正常模型不在列表中，使用第一个模型
                    eed_score, prompt_tokens, completion_tokens = self.evaluate_with_specific_model(self.model_keys[0])
                    self.history_rewards.append(eed_score)
                    self.total_delegations += 1
                    return eed_score, prompt_tokens, completion_tokens
            else:
                # 第二轮：策略模式
                if second_best_score == 0.0:
                    # 次优服务商无法解决，使用成本最低的模型（最后一个）
                    cheapest_model = self.model_keys[-1]
                    eed_score, prompt_tokens, completion_tokens = self.evaluate_with_specific_model(cheapest_model)
                    self.history_rewards.append(eed_score)
                    self.total_delegations += 1
                    return eed_score, prompt_tokens, completion_tokens
                else:
                    # 次优服务商能解决，从最差模型开始试，直到EED分数>=次优服务商
                    for idx in range(len(self.model_keys) - 1, -1, -1):  # 从最差到最好
                        model_key = self.model_keys[idx]
                        eed_score, prompt_tokens, completion_tokens = self.evaluate_with_specific_model(model_key)
                        
                        if eed_score >= second_best_score:
                            self.history_rewards.append(eed_score)
                            self.total_delegations += 1
                            return eed_score, prompt_tokens, completion_tokens
                    
                    # 如果所有模型都无法达到次优分数，使用最好的模型
                    best_model = self.model_keys[0]
                    eed_score, prompt_tokens, completion_tokens = self.evaluate_with_specific_model(best_model)
                    self.history_rewards.append(eed_score)
                    self.total_delegations += 1
                    return eed_score, prompt_tokens, completion_tokens
            
        except Exception as e:
            print(f"Provider {self.provider_id}: EED评估过程出错: {e}")
            self.history_rewards.append(0.0)
            self.total_delegations += 1
            return 0.0, 0, 0
    

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