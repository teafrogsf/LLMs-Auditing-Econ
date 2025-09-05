
### text_generation_model.py

**主要功能：**
- 支持多模型配置
- 动态定价策略实现
- 真实LLM调用和性能评估

**核心方法：**
```python
def set_cost(self, t: int, mechanism_info: Optional[Dict] = None) -> float:
    """设置当前时间步的成本，基于provider_id和历史数据"""

def get_price(self, t: int, mechanism_info: Optional[Dict] = None) -> float:
    """获取模型报价，用户可见的价格信息"""

def delegate_provider(self, phase: int, t: int, second_best_utility=None, R=None) -> Dict:
    """执行委托任务，返回评估结果
    
    策略说明：
    - 阶段1：使用真实成本模型
    - 阶段2：达到阈值后切换到最便宜模型
    - 阶段3：使用最便宜模型
    
    返回：
        {
            "reward": float,            # 评估分数
            "price": float,             # 价格（用户可见）
            "tokens": Tuple[int, int],  # (prompt_tokens, completion_tokens)
        }
    """
```

### user.py

**主要功能：**
- 多阶段机制执行
- 历史数据统计和分析
- 服务商性能评估

**核心方法：**
```python
def get_average_reward(self, provider_id: int) -> float:
    """获取指定服务商的历史平均奖励"""

def get_recent_average_reward(self, provider_id: int, recent_count: int) -> float:
    """获取指定服务商最近n次的平均奖励"""

def get_average_utility(self, provider_id: int) -> float:
    """获取指定服务商的历史平均效用"""

def get_recent_average_utility(self, provider_id: int, recent_count: int) -> float:
    """获取指定服务商最近n次的平均效用"""

def run_mechanism(self) -> Dict:
    """执行完整的三阶段机制"""
```

**机制参数：**
- `T`: 总时间步数
- `K`: 服务商数量
- `B = T^(2ε)`: 每阶段委托次数
- `M = T^(-ε)ln(KT)`: 机制参数
- `ε = 0.2`: 探索参数

### mechanism.py

**主要功能：**
- 阶段1：探索阶段，轮流委托每个服务商B次
- 阶段2：利用阶段，重点委托最优服务商，并对其他非最优服务商进行激励
- 阶段3：效用优化阶段，基于效用进行奖励

**核心方法：**
```python
def phase1_exploration(self, user):
    """阶段1：探索阶段，收集各服务商性能数据"""

def phase2_exploitation(self, user):
    """阶段2：利用阶段，重点使用最优服务商"""

def phase2_incentive(self, user):
    """阶段2：激励阶段，给予非最优服务商奖励"""

def phase3_utility_based(self, user):
    """阶段3：基于效用的最终奖励分配"""
```

### scenario.py - 场景配置

**主要功能：**
- 创建示例博弈场景
- 配置多个服务商的模型组合
- 设置博弈参数

## 输出结果

**运行结果包含：**
- `total_time`: 总时间步数
- `total_delegations`: 实际委托次数
- `best_provider_id`: 最优服务商ID
- `second_best_provider_id`: 次优服务商ID
- `avg_rewards`: 各服务商平均奖励
- `avg_utilities`: 各服务商平均效用
- `delegation_history`: 完整的委托历史记录

**文件输出：**
- 详细的博弈过程日志
- 各阶段的统计数据
- 服务商性能分析结果

## 注意事项

1. **模型定价**：基于真实API定价，支持动态成本计算
2. **并发处理**：机制支持多线程并发执行
3. **文件输出**：所有运行日志会保存到指定文件


## Uase by wangyu

```shell
python -
```