import math
import random
import sys
import numpy as np

from loguru import logger
from typing import List, Dict
from text_generation_model import Provider, MODEL_PRICING
from mechanism import Mechanism

class User:
    """委托人（用户）类"""

    def __init__(self, T: int, K: int, providers: List[Provider]):
        self.T = T  # 总时间步数
        self.K = K  # 服务商数量
        self.providers = providers

        # 机制参数
        self.epsilon = 0.2
        self.B = int(T ** (2 * self.epsilon))  # B = T^(2ϵ)
        self.M = (T ** (- self.epsilon)) * math.log(K * T)  # M = T^(-ϵ)ln(KT)

        # 博弈历史
        self.delegation_history = []  # 委托历史
        self.current_time = 0

        # 阶段记录
        self.phase1_completed = False
        self.phase2_completed_1 = False
        self.phase2_completed_2 = False
        self.phase3_completed = False

        # 阶段1结果
        self.avg_rewards = {}  # 各服务商平均回报
        self.avg_utilities = {}  # 各服务商平均utility
        self.best_provider = None
        self.second_best_reward = 0
        self.second_best_utility = 0
        self.second_best_provider = None
        
        # 历史回报记录 - 按服务商分组
        self.history_rewards = {}  # 按provider_id存储各服务商的历史回报
        self.history_utilities = {}  # 按provider_id存储各服务商的历史utility
        # 初始化每个服务商的历史回报和utility列表
        for provider in providers:
            self.history_rewards[provider.provider_id] = []
            self.history_utilities[provider.provider_id] = []
    
    def get_average_reward(self, provider_id: int) -> float:
        """获取指定服务商的历史平均回报"""
        if provider_id not in self.history_rewards or not self.history_rewards[provider_id]:
            return 0.0
        return float(np.mean(self.history_rewards[provider_id]))

    def get_recent_average_reward(self, provider_id: int, recent_count: int) -> float:
        """获取指定服务商最近n次的平均回报"""
        if provider_id not in self.history_rewards or not self.history_rewards[provider_id]:
            return 0.0
        recent_rewards = self.history_rewards[provider_id][-recent_count:]
        return float(np.mean(recent_rewards))
    
    def get_average_utility(self, provider_id: int) -> float:
        """获取指定服务商带来的的历史平均utility"""
        if provider_id not in self.history_utilities or not self.history_utilities[provider_id]:
            return 0.0
        return float(np.mean(self.history_utilities[provider_id]))

    def get_recent_average_utility(self, provider_id: int, recent_count: int) -> float:
        """获取指定服务商带来的最近n次的平均utility"""
        if provider_id not in self.history_utilities or not self.history_utilities[provider_id]:
            return 0.0
        recent_utilities = self.history_utilities[provider_id][-recent_count:]
        return float(np.mean(recent_utilities))
    
    # --------------------- 机制执行入口 --------------------- #
    def run_mechanism(self) -> Dict:
        """
        运行随机回报的效用保证机制
        Returns:
            博弈结果统计
        """

        self.mechanism = Mechanism()
        logger.info(f"开始运行机制，参数：T={self.T}, K={self.K}, B={self.B}, M={self.M:.4f}")

        # 阶段1：轮流委托每个服务商B次
        self._phase1_exploration()

        # 阶段2：委托最佳服务商
        self._phase2_exploitation()

        # 委托剩余服务商
        self._phase2_incentive()

        # 阶段3：基于效用的委托
        self._phase3_utility_based()

        return self._get_results()

    # --------------------- 阶段方法 --------------------- #
    def _phase1_exploration(self):
        """阶段1：探索阶段 - 通过mechanism委托"""
        self.mechanism.phase1_exploration(self)

    def _phase2_exploitation(self):
        """阶段2：利用阶段 - 通过mechanism委托"""
        self.mechanism.phase2_exploitation(self)

    def _phase2_incentive(self):
        """阶段3：激励阶段 - 通过mechanism委托"""
        self.mechanism.phase2_incentive(self)

    def _phase3_utility_based(self):
        """阶段4：基于效用的委托 - 通过mechanism委托"""
        self.mechanism.phase3_utility_based(self)

    def _get_results(self) -> Dict:
        """获取博弈结果统计"""
        results = {
            'total_time': self.T,
            'total_delegations': len(self.delegation_history),
            'phase1_completed': self.phase1_completed,
            'phase2_completed_1': self.phase2_completed_1,
            'phase2_completed_2': self.phase2_completed_2,
            'phase3_completed': self.phase3_completed,
            'best_provider': self.best_provider.provider_id if self.best_provider else None,
            'provider_stats': {}
        }

        # 统计每个服务商的表现
        for provider in self.providers:
            provider_delegations = [d for d in self.delegation_history if d['provider_id'] == provider.provider_id]

            if provider_delegations:
                total_cost = sum(d['cost'] for d in provider_delegations)
                total_reward = sum(d['reward'] for d in provider_delegations)
                avg_reward = total_reward / len(provider_delegations)
                total_prompt_tokens = sum(d.get('prompt_tokens', 0) for d in provider_delegations)
                total_completion_tokens = sum(d.get('completion_tokens', 0) for d in provider_delegations)
                total_tokens = sum(d.get('total_tokens', 0) for d in provider_delegations)
                results['provider_stats'][provider.provider_id] = {
                    'delegations': len(provider_delegations),
                    'total_cost': total_cost,
                    'total_reward': total_reward,
                    'avg_reward': avg_reward,  
                    'profit': total_reward - total_cost,
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_tokens': total_tokens,
                    'avg_tokens_per_delegation': total_tokens / len(provider_delegations) if len(provider_delegations) > 0 else 0
                }
            else:
                results['provider_stats'][provider.provider_id] = {
                    'delegations': 0,
                    'total_cost': 0,
                    'total_reward': 0,
                    'avg_reward': 0, 
                    'profit': 0,
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0,
                    'total_tokens': 0,
                    'avg_tokens_per_delegation': 0
                }

        return results
    