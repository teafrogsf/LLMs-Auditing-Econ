from ast import arg
import random
import argparse

import numpy as np
from loguru import logger

from text_generation_model import Provider, ProviderConfig
from user import User
import itertools


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
CHOICES=['honest', 'ours', 'worst', 'random']


def create_example_scenario(choices):
    # paser = argparse.ArgumentParser()
    # paser.add_argument('--s', help='strategy', default='ours', choices=['honest', 'ours', 'worst', 'random'])
    # args = paser.parse_args()
    logger.remove()
    logger.add(f"logs/simulator_{'-'.join(choices)}.log", rotation="10 MB", retention="7 days", level="INFO")

    T = 1000  # 总时间步数
    K = 3     # 服务商数量

    # 配置每个服务商的模型
    provider_settings = [
        dict(
            model_keys=["o1","o1-mini","gpt-4o-mini"],
        ),
        dict(
            model_keys=["o3-mini","deepseek-r1","deepseek-v3"],
        ),
        dict(
            model_keys=["gpt-4","gpt-4o","gpt-35-turbo"],
        )
    ]

    
    providers = []
    for i, setting in enumerate(provider_settings):
        config = ProviderConfig(
            provider_id=i + 1,
            price=0.0,
            model_keys=setting["model_keys"],
            model_costs=[],
            strategy=choices[i]
        )
        providers.append(Provider(config))

    # 创建User实例
    user = User(T, K, providers)

    results = user.run_mechanism()
    logger.info("\n=== 博弈结果 ===")
    logger.info(f"总时间步数：{results['total_time']}")
    logger.info(f"实际委托次数：{results['total_delegations']}")
    logger.info(f"最佳服务商：{results['best_provider']}")

    logger.info("\n各服务商统计：")
    for provider_id, stats in results['provider_stats'].items():
        # 获取对应provider的总成本
        provider = next(p for p in providers if p.provider_id == provider_id)
        total_cost = provider.get_total_cost()
        provider_utility = stats['total_price'] - total_cost  # 服务商效用 = price - cost
        
        logger.info(f"  服务商{provider_id}:")
        logger.info(f"    委托次数：{stats['delegations']}")
        logger.info(f"    总价格：{stats['total_price']:.4f}")
        logger.info(f"    总成本：{total_cost:.4f}")
        logger.info(f"    服务商效用：{provider_utility:.4f}")
        logger.info(f"    总回报：{stats['total_reward']:.4f}")
        logger.info(f"    平均回报：{stats['avg_reward']:.4f}")
        logger.info(f"    用户效用：{stats['profit']:.4f}")

    return user, providers, results

def main():
    choices_list = list(itertools.product(CHOICES, repeat=3))
    for item in choices_list:
        print(item)
        # continue
        create_example_scenario(item)


if __name__ == "__main__":
    # create_example_scenario()
    main()