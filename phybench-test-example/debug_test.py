#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试测试脚本 - 直接测试评估功能
"""

import sys
import os
import random
sys.path.append(os.path.abspath('../'))
from phybench_evaluation import PHYBenchEvaluator

def debug_test():
    """
    调试测试函数
    """
    print("开始调试测试...")
    
    # 创建评估器
    evaluator = PHYBenchEvaluator(models_to_test=["gpt-4o"])
    
    # 加载数据集
    print("加载数据集...")
    dataset = evaluator.load_dataset(None, None)  # 加载所有样本
    
    if not dataset:
        print("数据集加载失败")
        return
    
    
    # 使用test_random_sample函数测试随机样本
    try:
        model_name = "gpt-4o"
        print(f"测试模型: {model_name}")
        
        # 自动加载现有数据集，也可以传入上面的dataset
        eed_score = evaluator.test_random_sample(model_name, None)
        
        print("\n=== 随机样本测试结果 ===")
        print(f"模型名称: {model_name}")
        print(f"EED分数: {eed_score}")
        
        
    except Exception as e:
        import traceback
        print(f"随机样本测试失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_test()