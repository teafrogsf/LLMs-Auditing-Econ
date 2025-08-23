## 环境配置
创建并激活环境：
```bash
conda env create -p ./env_llm_auditing_econ -f environment.yml
conda activate ./env_llm_auditing_econ
```

## 项目结构

```
LLMs-Auditing-Econ/
├── benchmark/                    
│   └── nl_graph/                 # 图算法测试
│       ├── max_flow/             # 最大流问题
│       ├── shortest_path/        # 最短路径问题
│       ├── bipartite_matching/   # 二分图匹配问题
│       └── model_tester.py       # 模型测试器（用于测试benchmark在各个模型上的表现）
├── simulator/                    
│   ├── experiment.py             # 实验运行器
│   ├── mechanism.py              # 机制实现
│   ├── text_generation_model.py  # 文本生成模型（含策略实现）
│   ├── scenario.py               # 场景定义
│   └── user.py                   # 用户模拟
├── llm_client.py                 # LLM 客户端接口
├── model_evaluator.py            # 模型评估器（机制使用）
└── environment.yml               # 环境配置文件
```

## 使用方法

### 图算法基准测试

#### 使用 max_flow_evaluator.py 进行最大流测试

`max_flow_evaluator.py` 提供了一个模型的单独/多个最大流问题测试功能：

```python
from benchmark.nl_graph.max_flow.max_flow_evaluator import run_multi_graph_test, run_single_graph_test

# 运行单个随机图测试
single_result = run_single_graph_test("gpt-35-turbo")
print(f"单次测试得分: {single_result.get('score', 0)}")

# 运行多图测试（默认10个测试图），多线程进行（默认10线程）
multi_results = run_multi_graph_test("gpt-35-turbo", num_tests=10)
print(f"平均得分: {multi_results['average_score']}")
```

**更改测试模型：**
- 修改文件末尾的 `if __name__ == "__main__":`中`run_single_graph_test("gpt-35-turbo")` 部分中的模型名称，以字符串传入

#### 使用 model_tester.py 进行批量模型测试

`model_tester.py` 提供了多模型批量测试和结果可视化功能,运行结束后生成各模型得分平均分图表：

```python
from benchmark.nl_graph.model_tester import MultiModelRunner

# 创建测试运行器
runner = MultiModelRunner()

# 修改要测试的模型列表
runner.models = ['gpt-35-turbo', 'deepseek-v3']  # 添加你想测试的模型

# 运行所有模型测试
runner.run_all_tests()

# 生成结果图表
runner.plot_results()
```


**直接运行：**
```bash
python benchmark/nl_graph/model_tester.py
```

## 图算法模块详解

### 最大流问题 (max_flow)

- **max_flow_generator.py**: 生成最大流问题的测试图(单个图)
- **max_flow_dataset_generator.py**: 生成最大流问题的测试图数据集（1000个不相同）
- **max_flow_solver.py**: 构建提示词并评估LLM回答
- **max_flow_evaluator.py**: 提供完整的测试接口
- **max_flow_graphs.json**: 预生成的测试数据集
- **cot_prompt.txt**: Chain-of-Thought 提示模板

## 模型评估器接口

`model_evaluator.py` 提供统一的模型评估接口：

```python
def evaluate_model(model_name: str) -> Tuple[float, int, int]:
    """
    评估模型性能
    
    参数:
        model_name: 要测试的模型名称
    
    返回:
        Tuple[float, int, int]: (分数, input tokens数量, output tokens数量)
    """
```

## 机制模拟器

模拟器模块实现了机制的仿真环境，具体设计请参考：
- 论文中的机制设计部分
- `simulator/README.md` 中的详细说明

### 运行机制

```bash
python simulator/scenario.py
```

## 所有可用的模型

- gpt-4o, gpt-4, gpt-4o-mini, o1, o1-mini, o3-mini, gpt-35-turbo
- qwen-max
- deepseek-v3, deepseek-r1