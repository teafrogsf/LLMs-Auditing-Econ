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
            model_keys=["gpt-4","gpt-4o", "gpt-35-turbo"],
        ),
        # Qwen系列
        dict(
            model_keys=["qwen-max", "gpt-4o-mini"],
        ),
        # DeepSeek系列
        dict(
            model_keys=["deepseek-r1", "deepseek-v3"],
        ),
    ]

    
    providers = []
    for i, setting in enumerate(provider_settings):
        config = ProviderConfig(
            provider_id=i + 1,
            price=0.0,
            mu=0.1,  # 初始mu值设为默认值，将在第一轮后根据reward均值更新
            model_keys=setting["model_keys"],
            model_costs=[],
        )
        providers.append(Provider(config))

    # 创建User实例，指定输出文件
    user = User(T, K, providers, output_file="mechanism_output.txt")
    results = user.run_mechanism()

    return user, providers, results


if __name__ == "__main__":
    create_example_scenario()