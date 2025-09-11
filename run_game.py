import random
from tkinter import N
import numpy as np
import itertools
import json
import yaml
import argparse
import os
import math
from src.utils import Logger
from src.simulator import GameManager

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

    
    # 定义各服务商的η值
    eta_values = [0.2, 0.6, 0.4]  # provider1, provider2, provider3
    
    providers = []
    for i, setting in enumerate(provider_settings):
        config = ProviderConfig(
            provider_id=i + 1,
            price=0.0,
            model_keys=setting["model_keys"],
            model_costs=[],
            strategy=choices[i],
            eta=eta_values[i]
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
        provider_utility = provider.eta * stats['total_price'] - total_cost  # 服务商效用 = eta * price - cost
        
        logger.info(f"  服务商{provider_id}:")
        logger.info(f"    委托次数：{stats['delegations']}")
        logger.info(f"    总价格：{stats['total_price']:.4f}")
        logger.info(f"    总成本：{total_cost:.4f}")
        logger.info(f"    服务商效用：{provider_utility:.4f}")
        logger.info(f"    总回报：{stats['total_reward']:.4f}")
        logger.info(f"    平均回报：{stats['avg_reward']:.4f}")
        logger.info(f"    用户效用：{stats['profit']:.4f}")

    return user, providers, results


def init_config(path):
    config = yaml.safe_load(open(path))
    T = config['num_tasks']
    epsilon = config['epsilon']
    K = len(config['providers'])
    B = int(T ** (2 * epsilon))  # B = T^(2ϵ)
    M = (T ** (-epsilon)) * math.log(K * T)  # M = T^(-ϵ)ln(KT)
    config['B'] = B
    config['K'] = K
    config['M'] = M

    return config

def main():
    
    ### step1: load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='./config/default.yaml')
    parser.add_argument('--output-dir', type=str, default='./outputs/default/')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    ### step2: init logger and output dir
    # if not args.debug:
    #     if os.path.exists(args.output_dir):
    #         raise IOError(f'{args.output_dir} exists')
    #     os.makedirs(args.output_dir)
    # else:
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(name='llm-game', path=os.path.join(args.output_dir, 'log.txt'))
    
    


    ### step3: hyper para init
    config = init_config(args.config)
    logger.log('='*20+'Config'+"*"*20)
    logger.log(json.dumps(config, indent=2))

    ### step4: scenarios making
    scenarios = [('honest', 'worst', 'honest')]
    logger.log(f'{len(scenarios)}')
    for sc in scenarios:
        logger.log(f'run {sc}')
        config['output_dir'] = os.path.join(args.output_dir, "-".join(sc))
        for i in range(len(config['providers'])):
            config['providers'][i]['strategy'] = sc[i]
        game_manager = GameManager(config)
        game_manager.run_game()
    


if __name__ == "__main__":
    # create_example_scenario()
    main()