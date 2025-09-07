import math
import random
import numpy as np

from typing import Dict, List
# from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from text_generation_model import Provider


class Mechanism:
    """机制执行类，无状态的执行器，提供分阶段的方法供User调用"""

    def __init__(self):
        pass

    def phase1_exploration(self, user):
        """阶段1：轮流委托每个服务商B次"""
        logger.info(f"阶段1：轮流委托每个服务商{user.B}次")

        for provider in user.providers:
            logger.info(f"  委托服务商{provider.provider_id} {user.B}次")
            for _ in range(user.B):
                if user.current_time >= user.T:
                    break

                # 委托服务商，阶段1使用诚实模式
                result = provider.run(phase=1, t=user.current_time)
                reward = result["reward"]
                price = result["price"]
                prompt_tokens, completion_tokens = result["tokens"]

                user.delegation_history.append({
                    'time': user.current_time,
                    'provider_id': provider.provider_id,
                    'price': price,
                    'reward': reward,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                })
                
                # 记录该服务商的历史reward和utility
                utility = reward - price
                user.history_rewards[provider.provider_id].append(reward)
                user.history_utilities[provider.provider_id].append(utility)

                user.current_time += 1

        # 计算平均reward和平均utility
        for provider in user.providers:
            user.avg_rewards[provider.provider_id] = user.get_average_reward(provider.provider_id)
            user.avg_utilities[provider.provider_id] = user.get_average_utility(provider.provider_id)

        
        # 重新计算机制参数
        min_utility = min(user.avg_utilities.values())
        max_utility = max(user.avg_utilities.values())
        if min_utility <= 0:
            min_utility = 1e-6  # 使用一个很小的正数
            logger.warning(f"  警告：检测到utility值为0或负数，使用默认值 {min_utility}")
        user.delta_1 = -math.log(min_utility) + 2 + user.M  # δ1 = -log(min_i utility_i) + 2 + M
        user.delta_2 = math.log(max_utility) # δ2 = log(max_i utility_i)
        logger.info(f"  更新后的值：δ1={user.delta_1:.4f}, δ2={user.delta_2:.4f}")

        # 找到最佳服务商（基于utility）
        best_utility = max(user.avg_utilities.values())
        best_providers = [p for p in user.providers if user.avg_utilities[p.provider_id] == best_utility]
        user.best_provider = random.choice(best_providers)

        # 找到第二好的utility
        utilities = list(user.avg_utilities.values())
        utilities.remove(best_utility)
        user.second_best_utility = max(utilities) if utilities else 0
        user.second_best_provider = max(
            [p for p in user.providers if user.avg_utilities[p.provider_id] == user.second_best_utility],
            key=lambda p: p.provider_id
        ) if utilities else None
        
        logger.info(f"  阶段1完成，最佳服务商：{user.best_provider.provider_id if user.best_provider else None}，平均效用：{best_utility:.4f}")
        logger.info(f"  第二好效用：{user.second_best_utility:.4f}")

        user.phase1_completed = True

    def phase2_exploitation(self, user):
        """阶段2：委托最佳服务商，采用新策略"""
        if not user.phase1_completed:
            return

        if user.best_provider is None:
            user.phase2_completed = True
            return

        logger.info(f"阶段2：委托最佳服务商{user.best_provider.provider_id}")

        threshold = user.second_best_utility - user.M
        R = max(0, user.T - (max(user.delta_1,user.delta_2) + 3) * user.B * user.K)
        remaining_delegations = int(min(R, user.T - user.current_time))

        logger.info(f"  计划委托{remaining_delegations}次，阈值：{threshold:.4f}")

        delegation_count = 0
        stopped_early = False

        # 顺序执行委托任务
        for i in range(remaining_delegations):
            if user.current_time >= user.T:
                break
                
            # 执行委托任务
            result = user.best_provider.run(phase=2, t=user.current_time, second_best_utility=user.second_best_utility, R=remaining_delegations)
            reward = result["reward"]
            price = result["price"]
            prompt_tokens, completion_tokens = result["tokens"]
            
            user.delegation_history.append({
                'time': user.current_time,
                'provider_id': user.best_provider.provider_id,
                'price': price,
                'reward': reward,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            })
            
            # 记录该服务商的历史回报和utility
            utility = reward - price
            user.history_rewards[user.best_provider.provider_id].append(reward)
            user.history_utilities[user.best_provider.provider_id].append(utility)
            
            delegation_count = i + 1
            user.current_time += 1
            
            # 检查是否需要提前停止
            if delegation_count >= user.B and user.best_provider is not None:
                recent_avg_utility = user.get_recent_average_utility(user.best_provider.provider_id, user.B)
                if recent_avg_utility < threshold:
                    logger.info(f"  在{delegation_count}次委托后停止，最近平均utility：{recent_avg_utility:.4f} < {threshold:.4f}")
                    stopped_early = True
                    break

        # 如果没有提前停止，给予奖励
        if not stopped_early and user.current_time < user.T and user.best_provider is not None:
            logger.info(f"  给予奖励，额外委托{user.B}次")
            for _ in range(min(user.B, user.T - user.current_time)):
                # 奖励轮委托
                result = user.best_provider.run(phase=3, t=user.current_time, second_best_utility=user.second_best_utility)
                reward = result["reward"]
                price = result["price"]
                prompt_tokens, completion_tokens = result["tokens"]

                user.delegation_history.append({
                    'time': user.current_time,
                    'provider_id': user.best_provider.provider_id,
                    'price': price,
                    'reward': reward,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                })
                
                # 记录该服务商的历史回报和utility
                utility = reward - price
                user.history_rewards[user.best_provider.provider_id].append(reward)
                user.history_utilities[user.best_provider.provider_id].append(utility)

                user.current_time += 1

        user.phase2_completed_1 = True

    def phase2_incentive(self, user):
        """阶段2：委托除最优服务商外的其他服务商各B次"""
        if not user.phase2_completed_1:
            return

        logger.info("阶段2：委托除最优服务商外的其他服务商")

        # 委托除最优服务商外的其他服务商
        for provider in user.providers:
            if user.current_time >= user.T:
                break
                
            # 跳过最优服务商
            if user.best_provider and provider.provider_id == user.best_provider.provider_id:
                continue
                
            logger.info(f"  委托服务商{provider.provider_id} {user.B}次")
            
            # 委托该服务商B次
            for _ in range(user.B):
                result = provider.run(phase=3, t=user.current_time)
                reward = result["reward"]
                price = result["price"]
                prompt_tokens, completion_tokens = result["tokens"]

                user.delegation_history.append({
                    'time': user.current_time,
                    'provider_id': provider.provider_id,
                    'price': price,
                    'reward': reward,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens
                })
                
                # 记录该服务商的历史回报和utility
                utility = reward - price
                user.history_rewards[provider.provider_id].append(reward)
                user.history_utilities[provider.provider_id].append(utility)

                user.current_time += 1

        user.phase2_completed_2 = True

    def phase3_utility_based(self, user):
        """阶段3：基于效用的委托"""
        if not user.phase2_completed_2:
            return

        logger.info("阶段3：基于效用的委托")

        for provider in user.providers:
            if user.current_time >= user.T:
                break

            avg_reward = user.avg_rewards[provider.provider_id]
            # 确保对数为正数
            reward_diff = avg_reward - user.M
            if reward_diff <= 0:
                logger.info(f"  服务商{provider.provider_id}的回报差值{reward_diff:.4f} <= 0，跳过")
                continue

            utility = user.delta_1 + math.log(reward_diff)

            if utility >= 0:
                # 计算委托次数
                integer_part = int(utility)
                fractional_part = utility - integer_part

                logger.info(f"  服务商{provider.provider_id}效用：{utility:.4f}，整数部分：{integer_part}，小数部分：{fractional_part:.4f}")

                # 整数部分委托
                integer_delegations = integer_part * user.B
                for _ in range(min(integer_delegations, user.T - user.current_time)):
                    # 整数部分委托
                    result = provider.run(phase=4, t=user.current_time)
                    reward = result["reward"]
                    price = result["price"]
                    prompt_tokens, completion_tokens = result["tokens"]

                    user.delegation_history.append({
                        'time': user.current_time,
                        'provider_id': provider.provider_id,
                        'price': price,
                        'reward': reward,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': prompt_tokens + completion_tokens
                    })
                    
                    # 记录该服务商的历史回报和utility
                    utility_value = reward - price
                    user.history_rewards[provider.provider_id].append(reward)
                    user.history_utilities[provider.provider_id].append(utility_value)

                    user.current_time += 1

                # 小数部分概率委托
                if fractional_part > 0 and user.current_time < user.T:
                    if random.random() < fractional_part:
                        logger.info(f"  服务商{provider.provider_id}获得概率委托，概率：{fractional_part:.4f}")
                        for _ in range(min(user.B, user.T - user.current_time)):
                            # 概率委托
                            result = provider.run(phase=4, t=user.current_time)
                            reward = result["reward"]
                            price = result["price"]
                            prompt_tokens, completion_tokens = result["tokens"]

                            user.delegation_history.append({
                                'time': user.current_time,
                                'provider_id': provider.provider_id,
                                'price': price,
                                'reward': reward,
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': prompt_tokens + completion_tokens
                            })
                            
                            # 记录该服务商的历史回报和utility
                            utility_value = reward - price
                            user.history_rewards[provider.provider_id].append(reward)
                            user.history_utilities[provider.provider_id].append(utility_value)

                            user.current_time += 1

        user.phase3_completed = True