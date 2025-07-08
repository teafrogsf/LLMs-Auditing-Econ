import math
import random
from typing import List, Dict

import numpy as np

from provider import Provider, MODEL_PRICING
from twenty_four_game import generate_hard_24_problem, check_24_answer


class User:
    """委托人（用户）类"""

    def __init__(self, T: int, K: int, providers: List[Provider]):
        self.T = T  # 总时间步数
        self.K = K  # 服务商数量
        self.providers = providers

        # 机制参数
        self.B = int(T ** (2 / 3))  # B = T^(2/3)
        self.M = 8 * (T ** (-1 / 3)) * math.log(K * T)  # M = 8T^(-1/3)ln(KT)
        self.u = 1 + self.M

        # 博弈历史
        self.delegation_history = []  # 委托历史
        self.current_time = 0

        # 阶段记录
        self.phase1_completed = False
        self.phase2_completed = False
        self.phase3_completed = False

        # 阶段1结果
        self.avg_rewards = {}  # 各服务商平均回报
        self.best_provider = None
        self.second_best_reward = 0
        self.second_best_provider = None

    # --------------------- 机制执行入口 --------------------- #
    def run_mechanism(self) -> Dict:
        """
        运行随机回报的效用保证机制
        Returns:
            博弈结果统计
        """
        print(f"开始运行机制，参数：T={self.T}, K={self.K}, B={self.B}, M={self.M:.4f}, u={self.u:.4f}")

        # 阶段1：轮流委托每个服务商B次
        self._phase1_exploration()

        # 阶段2：委托最佳服务商
        self._phase2_exploitation()

        # 阶段3：基于效用的委托
        self._phase3_utility_based()

        return self._get_results()

    # --------------------- 阶段 1 --------------------- #
    def _phase1_exploration(self):
        """阶段1：轮流委托每个服务商B次"""
        print(f"阶段1：轮流委托每个服务商{self.B}次")

        for provider in self.providers:
            print(f"  委托服务商{provider.provider_id} {self.B}次")
            for _ in range(self.B):
                if self.current_time >= self.T:
                    break

                # 委托服务商
                cost = provider.set_cost(self.current_time)
                reward = provider.generate_reward(cost)

                self.delegation_history.append({
                    'time': self.current_time,
                    'provider_id': provider.provider_id,
                    'cost': cost,
                    'reward': reward
                })

                self.current_time += 1

        # 计算平均回报
        for provider in self.providers:
            self.avg_rewards[provider.provider_id] = provider.get_average_reward()

        # 找到最佳服务商
        best_reward = max(self.avg_rewards.values())
        self.best_provider = max(
            [p for p in self.providers if self.avg_rewards[p.provider_id] == best_reward],
            key=lambda p: p.provider_id
        )

        # 找到第二好的回报
        rewards = list(self.avg_rewards.values())
        rewards.remove(best_reward)
        self.second_best_reward = max(rewards) if rewards else 0
        self.second_best_provider = max(
            [p for p in self.providers if self.avg_rewards[p.provider_id] == self.second_best_reward],
            key=lambda p: p.provider_id
        ) if rewards else None

        print(f"  阶段1完成，最佳服务商：{self.best_provider.provider_id if self.best_provider else None}，平均回报：{best_reward:.4f}")
        print(f"  第二好回报：{self.second_best_reward:.4f}")

        self.phase1_completed = True

    # --------------------- 阶段 2 --------------------- #
    def _phase2_exploitation(self):
        """阶段2：委托最佳服务商，采用新策略"""
        if not self.phase1_completed:
            return

        if self.best_provider is None:
            self.phase2_completed = True
            return

        print(f"阶段2：委托最佳服务商{self.best_provider.provider_id}")

        threshold = self.second_best_reward - self.M
        R = max(0, self.T - (self.u + 3) * self.B * self.K)
        remaining_delegations = min(R, self.T - self.current_time)

        print(f"  计划委托{remaining_delegations}次，阈值：{threshold:.4f}")

        delegation_count = 0
        stopped_early = False

        while delegation_count < remaining_delegations and self.current_time < self.T:
            nums = generate_hard_24_problem()
            can_solve_2 = self.second_best_provider.solve_24(nums) if self.second_best_provider else None
            if can_solve_2:
                expr = self.best_provider.solve_24(nums)
                cost = self.best_provider.set_cost(self.current_time)
                reward = 1.0 if expr else 0.0
            else:
                # 用best_provider最差模型
                idx = len(self.best_provider.llms) - 1
                llm = self.best_provider.llms[idx]
                try:
                    answer, prompt_tokens, completion_tokens = llm.call_llm(
                        f"请用加减乘除和括号将{nums}这四个数拼成24，要求每个数只能用一次，返回一个合法表达式。只返回表达式本身。"
                    )
                    expr = answer.strip().split('\n')[0]
                    self.best_provider.last_model_idx = idx
                    self.best_provider.last_tokens = (prompt_tokens, completion_tokens)
                    cost = self.best_provider.set_cost(self.current_time)
                    reward = 1.0 if check_24_answer(nums, expr) else 0.0
                except Exception:
                    cost = self.best_provider.price
                    reward = 0.0
            self.delegation_history.append({
                'time': self.current_time,
                'provider_id': self.best_provider.provider_id,
                'cost': cost,
                'reward': reward
            })
            delegation_count += 1
            self.current_time += 1
            if delegation_count >= self.B and self.best_provider is not None:
                recent_avg = self.best_provider.get_recent_average_reward(self.B)
                if recent_avg < threshold:
                    print(f"  在{delegation_count}次委托后停止，最近平均回报：{recent_avg:.4f} < {threshold:.4f}")
                    stopped_early = True
                    break

        # 如果没有提前停止，给予奖励
        if not stopped_early and self.current_time < self.T and self.best_provider is not None:
            print(f"  给予奖励，额外委托{self.B}次")
            for _ in range(min(self.B, self.T - self.current_time)):
                cost = self.best_provider.set_cost(self.current_time, {
                    'threshold': threshold,
                    'phase': 2,
                    'reward': True
                })
                reward = self.best_provider.generate_reward(cost)

                self.delegation_history.append({
                    'time': self.current_time,
                    'provider_id': self.best_provider.provider_id,
                    'cost': cost,
                    'reward': reward
                })

                self.current_time += 1

        self.phase2_completed = True

    # --------------------- 阶段 3 --------------------- #
    def _phase3_utility_based(self):
        """阶段3：基于效用的委托"""
        if not self.phase2_completed:
            return

        print("阶段3：基于效用的委托")

        for provider in self.providers:
            if self.current_time >= self.T:
                break

            avg_reward = self.avg_rewards[provider.provider_id]
            # 确保对数为正数
            reward_diff = avg_reward - self.M
            if reward_diff <= 0:
                print(f"  服务商{provider.provider_id}的回报差值{reward_diff:.4f} <= 0，跳过")
                continue

            utility = self.u + math.log(reward_diff)

            if utility >= 0:
                # 计算委托次数
                integer_part = int(utility)
                fractional_part = utility - integer_part

                print(f"  服务商{provider.provider_id}效用：{utility:.4f}，整数部分：{integer_part}，小数部分：{fractional_part:.4f}")

                # 整数部分委托
                integer_delegations = integer_part * self.B
                for _ in range(min(integer_delegations, self.T - self.current_time)):
                    cost = provider.set_cost(self.current_time, {
                        'utility': utility,
                        'phase': 3
                    })
                    reward = provider.generate_reward(cost)

                    self.delegation_history.append({
                        'time': self.current_time,
                        'provider_id': provider.provider_id,
                        'cost': cost,
                        'reward': reward
                    })

                    self.current_time += 1

                # 小数部分概率委托
                if fractional_part > 0 and self.current_time < self.T:
                    if random.random() < fractional_part:
                        print(f"  服务商{provider.provider_id}获得概率委托，概率：{fractional_part:.4f}")
                        for _ in range(min(self.B, self.T - self.current_time)):
                            cost = provider.set_cost(self.current_time, {
                                'utility': utility,
                                'phase': 3,
                                'probabilistic': True
                            })
                            reward = provider.generate_reward(cost)

                            self.delegation_history.append({
                                'time': self.current_time,
                                'provider_id': provider.provider_id,
                                'cost': cost,
                                'reward': reward
                            })

                            self.current_time += 1

        self.phase3_completed = True

    # --------------------- 结果统计 --------------------- #
    def _get_results(self) -> Dict:
        """获取博弈结果统计"""
        results = {
            'total_time': self.T,
            'total_delegations': len(self.delegation_history),
            'phase1_completed': self.phase1_completed,
            'phase2_completed': self.phase2_completed,
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

                results['provider_stats'][provider.provider_id] = {
                    'delegations': len(provider_delegations),
                    'total_cost': total_cost,
                    'total_reward': total_reward,
                    'avg_reward': avg_reward,
                    'profit': total_reward - total_cost
                }
            else:
                results['provider_stats'][provider.provider_id] = {
                    'delegations': 0,
                    'total_cost': 0,
                    'total_reward': 0,
                    'avg_reward': 0,
                    'profit': 0
                }

        return results 