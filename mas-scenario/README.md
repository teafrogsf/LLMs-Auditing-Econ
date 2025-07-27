## 文档说明
# `utils.py`
提供接口`def evaluate_model(self, model_name: str, test_sample: Dict) -> Tuple[float, int, int]`,参数为要测试的模型名称，以及test sample，返回该模型此次测试得分，input tokens数量以及output tokens数量

# `provider.py`
1. `def set_cost(self, t: int, mechanism_info: Optional[Dict] = None) -> float:` 传入当前时间步t，以及历史真实使用模型列表，返回当前花费的成本c.（这里根据provider_id的不同有不同的η，从price得出cost）
2. `def delegate_provider(self, phase: int, test_sample: Dict，second_best_reward=None) -> Dict:`产生reward的函数，内部调用`evaluate_model`函数以及`get_price`函数，具有不同策略.阶段一时（t处于1-B）永远使用真实成本；阶段二时首先使用真实成本，当产生的reward累积达到R*second_best_reward时，使用最便宜的模型；最后奖励轮次（若有），调用最便宜的模型.返回：
            Dict:
                {
                    "reward": float,            # 评估分数
                    "price": float,              # 使用该模型产生的价格（用户可见）
                    "tokens": Tuple[int, int],  # (prompt_token, completion_token)
                }
（手动在调用cost之前添加真实使用模型列表以及使用的tokens信息）

# `user.py`
1. 初始化数据集，进行随机抽取test sample.
`def get_average_reward(self) -> float:` 获取历史平均reawrd
`def get_recent_average_reward(self, recent_count: int) -> float:`获取最近n次的平均回报
2. 阶段一，每个服务商分配B次，计算平均回报，查找最优以及次优服务商（这一轮结束的时候计算出各服务商的𝜇_i）
3. 阶段二，最优服务商调用R次，若没有提前终止，额外奖励B次
4. 对于非最优服务商，奖励B次
5. 对所有满足条件的服务商再进行额外奖励和概率奖励

我们希望实验的结果是profit比较好看（？）