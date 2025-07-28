import math
import random
import sys
from typing import List, Dict

import numpy as np

from provider import Provider, MODEL_PRICING


class User:
    """委托人（用户）类"""

    def __init__(self, T: int, K: int, providers: List[Provider], output_file: str = "output.txt"):
        self.T = T  # 总时间步数
        self.K = K  # 服务商数量
        self.providers = providers
        
        # 输出文件设置
        self.output_file = output_file
        self.original_stdout = sys.stdout
        
        # 机制参数
        self.B = int(T ** (2 / 3))  # B = T^(2/3)
        self.M = 8 * (T ** (-1 / 3)) * math.log(K * T)  # M = 8T^(-1/3)ln(KT)
        min_mu = min(p.mu for p in providers)
        self.u = -math.log(min_mu) + 1 + self.M  # u = -log(min_i μ_i) + 1 + M

        # 博弈历史
        self.delegation_history = []  # 委托历史
        self.current_time = 0

        # 阶段记录
        self.phase1_completed = False
        self.phase2_completed = False
        self.phase3_completed = False
        self.phase4_completed = False

        # 阶段1结果
        self.avg_rewards = {}  # 各服务商平均回报
        self.best_provider = None
        self.second_best_reward = 0
        self.second_best_provider = None
        
        # 历史回报记录 - 按服务商分组
        self.history_rewards = {}  # 按provider_id存储各服务商的历史回报
        # 初始化每个服务商的历史回报列表
        for provider in providers:
            self.history_rewards[provider.provider_id] = []
    
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
    
    def _print_to_file(self, *args, **kwargs):
        """将输出重定向到文件"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            print(*args, file=f, **kwargs)
    
    def _start_file_output(self):
        """开始将输出重定向到文件"""
        # 清空输出文件
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("=== 机制执行日志 ===\n\n")
        # 重定向标准输出，设置为无缓冲模式实现实时输出
        sys.stdout = open(self.output_file, 'a', encoding='utf-8', buffering=1)
    
    def _stop_file_output(self):
        """停止输出重定向，恢复到控制台"""
        if sys.stdout != self.original_stdout:
            sys.stdout.close()
            sys.stdout = self.original_stdout

    # --------------------- 机制执行入口 --------------------- #
    def run_mechanism(self) -> Dict:
        """
        运行随机回报的效用保证机制
        Returns:
            博弈结果统计
        """
        # 开始将输出重定向到文件
        self._start_file_output()
        
        try:
            print(f"开始运行机制，参数：T={self.T}, K={self.K}, B={self.B}, M={self.M:.4f}, u={self.u:.4f}")

            # 阶段1：轮流委托每个服务商B次
            self._phase1_exploration()

            # 阶段2：委托最佳服务商
            self._phase2_exploitation()

            # 阶段3：委托剩余服务商
            self._phase3_incentive()

            # 阶段4：基于效用的委托
            self._phase4_utility_based()

            results = self._get_results()
            
            # 在文件中输出最终结果
            print("\n=== 博弈结果 ===")
            print(f"总时间步数：{results['total_time']}")
            print(f"实际委托次数：{results['total_delegations']}")
            print(f"最佳服务商：{results['best_provider']}")

            print("\n各服务商统计：")
            for provider_id, stats in results['provider_stats'].items():
                print(f"  服务商{provider_id}:")
                print(f"    委托次数：{stats['delegations']}")
                print(f"    总成本：{stats['total_cost']:.4f}")
                print(f"    总回报：{stats['total_reward']:.4f}")
                print(f"    平均回报：{stats['avg_reward']:.4f}")
                print(f"    用户效用：{stats['profit']:.4f}")
            
            return results
        
        finally:
            # 恢复标准输出
            self._stop_file_output()
            # 在控制台显示完成信息
            print(f"机制执行完成，详细日志已保存到: {self.output_file}")

    # --------------------- 阶段 1 --------------------- #
    def _phase1_exploration(self):
        """阶段1：轮流委托每个服务商B次"""
        print(f"阶段1：轮流委托每个服务商{self.B}次")

        for provider in self.providers:
            print(f"  委托服务商{provider.provider_id} {self.B}次")
            for _ in range(self.B):
                if self.current_time >= self.T:
                    break

                # 委托服务商，阶段1使用诚实模式
                result = provider.delegate_provider(phase=1, t=self.current_time)
                reward = result["reward"]
                price = result["price"]
                prompt_tokens, completion_tokens = result["tokens"]

                self.delegation_history.append({
                    'time': self.current_time,
                    'provider_id': provider.provider_id,
                    'cost': price,
                    'reward': reward,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                })
                
                # 记录该服务商的历史回报
                self.history_rewards[provider.provider_id].append(reward)

                self.current_time += 1

        # 计算平均回报
        for provider in self.providers:
            self.avg_rewards[provider.provider_id] = self.get_average_reward(provider.provider_id)

        # 第一轮后更新各服务商的mu值
        print("\n  第一轮完成，更新各服务商的mu值：")
        for provider in self.providers:
            provider.update_mu_from_rewards(self)
        
        # 重新计算机制参数，因为mu值已更新
        min_mu = min(p.mu for p in self.providers)
        if min_mu <= 0:
            min_mu = 1e-6  # 使用一个很小的正数
            print(f"  警告：检测到mu值为0或负数，使用默认值 {min_mu}")
        self.u = -math.log(min_mu) + 1 + self.M  # u = -log(min_i μ_i) + 1 + M
        print(f"  更新后的u值：{self.u:.4f}")

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
            # 阶段2委托最优服务商
            result = self.best_provider.delegate_provider(phase=2, t=self.current_time, second_best_reward=self.second_best_reward, R=remaining_delegations)
            reward = result["reward"]
            price = result["price"]
            prompt_tokens, completion_tokens = result["tokens"]
            
            self.delegation_history.append({
                'time': self.current_time,
                'provider_id': self.best_provider.provider_id,
                'cost': price,
                'reward': reward,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            })
            
            # 记录该服务商的历史回报
            self.history_rewards[self.best_provider.provider_id].append(reward)
            
            delegation_count += 1
            self.current_time += 1
            if delegation_count >= self.B and self.best_provider is not None:
                recent_avg = self.get_recent_average_reward(self.best_provider.provider_id, self.B)
                if recent_avg < threshold:
                    print(f"  在{delegation_count}次委托后停止，最近平均回报：{recent_avg:.4f} < {threshold:.4f}")
                    stopped_early = True
                    break

        # 如果没有提前停止，给予奖励
        if not stopped_early and self.current_time < self.T and self.best_provider is not None:
            print(f"  给予奖励，额外委托{self.B}次")
            for _ in range(min(self.B, self.T - self.current_time)):
                # 奖励轮委托
                result = self.best_provider.delegate_provider(phase=3, t=self.current_time, second_best_reward=self.second_best_reward)
                reward = result["reward"]
                price = result["price"]
                prompt_tokens, completion_tokens = result["tokens"]

                self.delegation_history.append({
                    'time': self.current_time,
                    'provider_id': self.best_provider.provider_id,
                    'cost': price,
                    'reward': reward,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                })
                
                # 记录该服务商的历史回报
                self.history_rewards[self.best_provider.provider_id].append(reward)

                self.current_time += 1

        self.phase2_completed = True
    # --------------------- 阶段 3 --------------------- #
    def _phase3_incentive(self):
        """阶段3：委托除最优服务商外的其他服务商各B次"""
        if not self.phase2_completed:
            return

        print("阶段3：委托除最优服务商外的其他服务商")

        # 委托除最优服务商外的其他服务商
        for provider in self.providers:
            if self.current_time >= self.T:
                break
                
            # 跳过最优服务商
            if self.best_provider and provider.provider_id == self.best_provider.provider_id:
                continue
                
            print(f"  委托服务商{provider.provider_id} {self.B}次")
            
            # 委托该服务商B次
            for _ in range(min(self.B, self.T - self.current_time)):#DONE: 这里的委托次数需要调整
                # 阶段3委托
                result = provider.delegate_provider(phase=3, t=self.current_time)
                reward = result["reward"]
                price = result["price"]
                prompt_tokens, completion_tokens = result["tokens"]

                self.delegation_history.append({
                    'time': self.current_time,
                    'provider_id': provider.provider_id,
                    'cost': price,
                    'reward': reward,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                })
                
                # 记录该服务商的历史回报
                self.history_rewards[provider.provider_id].append(reward)

                self.current_time += 1

        self.phase3_completed = True

    # --------------------- 阶段 4 --------------------- #
    def _phase4_utility_based(self):
        """阶段4：基于效用的委托"""
        if not self.phase2_completed:
            return

        print("阶段4：基于效用的委托")

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
                    # 阶段4整数部分委托
                    result = provider.delegate_provider(phase=4, t=self.current_time)
                    reward = result["reward"]
                    price = result["price"]
                    prompt_tokens, completion_tokens = result["tokens"]

                    self.delegation_history.append({
                        'time': self.current_time,
                        'provider_id': provider.provider_id,
                        'cost': price,
                        'reward': reward,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': prompt_tokens + completion_tokens
                    })
                    
                    # 记录该服务商的历史回报
                    self.history_rewards[provider.provider_id].append(reward)

                    self.current_time += 1

                # 小数部分概率委托
                if fractional_part > 0 and self.current_time < self.T:
                    if random.random() < fractional_part:
                        print(f"  服务商{provider.provider_id}获得概率委托，概率：{fractional_part:.4f}")
                        for _ in range(min(self.B, self.T - self.current_time)):
                            # 阶段4概率委托
                            result = provider.delegate_provider(phase=4, t=self.current_time)
                            reward = result["reward"]
                            price = result["price"]
                            prompt_tokens, completion_tokens = result["tokens"]

                            self.delegation_history.append({
                                'time': self.current_time,
                                'provider_id': provider.provider_id,
                                'cost': price,
                                'reward': reward,
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': prompt_tokens + completion_tokens
                            })
                            
                            # 记录该服务商的历史回报
                            self.history_rewards[provider.provider_id].append(reward)

                            self.current_time += 1

        self.phase4_completed = True

    # --------------------- 结果统计 --------------------- #
    def _get_results(self) -> Dict:
        """获取博弈结果统计"""
        results = {
            'total_time': self.T,
            'total_delegations': len(self.delegation_history),
            'phase1_completed': self.phase1_completed,
            'phase2_completed': self.phase2_completed,
            'phase3_completed': self.phase3_completed,
            'phase4_completed': self.phase4_completed,
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
                    'avg_reward': avg_reward,  # avg_reward即为平均EED分数
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
                    'avg_reward': 0,  # avg_reward即为平均EED分数
                    'profit': 0,
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0,
                    'total_tokens': 0,
                    'avg_tokens_per_delegation': 0
                }

        return results
    