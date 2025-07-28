import random
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_client import ExampleLLM

# 添加phybench-test目录到路径以导入EED评分功能
sys.path.append(os.path.join(os.path.dirname(__file__), 'phybench-test-example'))
from phybench_evaluation import PHYBenchEvaluator

evaluator = PHYBenchEvaluator()
dataset = evaluator.load_dataset(None, None)
def evaluate_model(model_name: str) -> Tuple[float, int, int]:
    """使用指定模型在数据集中随机选择样本进行评估
    
    Args:
        model_name: 要测试的模型名称
        
    Returns:
        Tuple[float, int, int]: (分数, input tokens数量, output tokens数量)
    """
    if not dataset:
        return 0.0, 0, 0
    
    # 随机选择一个样本
    sample = random.choice(dataset)
    
    # 创建模型
    model = ExampleLLM(model_name)
    
    # 调用evaluate_single_sample函数
    result = evaluator.evaluate_single_sample(model, sample)
    
    # 提取返回值
    score = result.get('eed_score', 0.0)
    prompt_tokens = result.get('prompt_tokens', 0)
    completion_tokens = result.get('completion_tokens', 0)
    
    return float(score), int(prompt_tokens), int(completion_tokens)
    