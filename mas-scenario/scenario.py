import random

import numpy as np

from provider import Provider, ProviderConfig
from user import User


def create_example_scenario():
    np.random.seed(42)
    random.seed(42)

    T = 10  # 总时间步数
    K = 3     # 服务商数量

    # 配置每个服务商的模型
    provider_settings = [
        # GPT系列
        dict(
            model_keys=["o1","o3-mini", "o1-mini"],
        ),
        # Qwen系列
        dict(
            model_keys=["deepseek-r1", "gpt-4o-mini","deepseek-v3"],
        ),
        # DeepSeek系列
        dict(
            model_keys=["qwen-max", "gpt-4o", "gpt-35-turbo"],
        ),
    ]

    
    providers = []
    for i, setting in enumerate(provider_settings):
        config = ProviderConfig(
            provider_id=i + 1,
            price=0.0,
            mu=5,  # 初始mu值设为默认值，将在第一轮后根据reward均值更新
            model_keys=setting["model_keys"],
            model_costs=[],
        )
        providers.append(Provider(config))

    # 创建User实例，指定输出文件
    user = User(T, K, providers, output_file="output.txt")

    results = user.run_mechanism()
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

    return user, providers, results


if __name__ == "__main__":
    create_example_scenario()