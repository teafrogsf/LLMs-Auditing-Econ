import random
import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_client import ExampleLLM

sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark", "nl_graph", "max_flow"))
from max_flow_evaluator import generate_and_evaluate_random_graph

def evaluate_model(model_name: str) -> Tuple[float, int, int]:
    """使用指定模型在数据集中随机选择样本进行评估
    
    Args:
        model_name: 要测试的模型名称
        
    Returns:
        Tuple[float, int, int]: (分数, input tokens数量, output tokens数量)
    """
    result = generate_and_evaluate_random_graph(model_name)
    
    # 提取返回值
    score = result.get('score', 0.0) * 10 # reward放大10倍
    prompt_tokens = result.get('prompt_tokens', 0)
    completion_tokens = result.get('completion_tokens', 0)
    
    return score, prompt_tokens, completion_tokens
    