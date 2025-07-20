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
    evaluator = PHYBenchEvaluator(models_to_test=["o1-mini"])
    
    # 加载数据集
    print("加载数据集...")
    dataset = evaluator.load_dataset(None, None)  # 加载所有样本
    
    if not dataset:
        print("数据集加载失败")
        return
    
    
    # 使用evaluate_single_sample函数测试随机样本并打印模型回答
    try:
        model_name = "o1-mini"
        print(f"测试模型: {model_name}")
        
        # 随机选择一个样本
        import random
        selected_sample = random.choice(dataset)
        print(f"\n选择的样本ID: {selected_sample.get('id', 'N/A')}")
        
        # 初始化模型
        from llm_client import ExampleLLM
        model = ExampleLLM(model_name)
        
        # 评估单个样本
        result = evaluator.evaluate_single_sample(model, selected_sample)
        
        print("\n=== 调试测试结果 ===")
        print(f"模型名称: {model_name}")
        print(f"问题: {result.get('problem', 'N/A')}")
        print(f"\n模型完整回答:\n{result.get('full_response', 'N/A')}")
        print(f"\n提取的预测答案: {result.get('predicted_answer', 'N/A')}")
        print(f"标准答案: {result.get('ground_truth', 'N/A')}")
        print(f"EED分数: {result.get('eed_score', 0.0)}")
        print(f"相对距离: {result.get('relative_distance', -1)}")
        print(f"评估状态: {'成功' if result.get('success', False) else '失败'}")
        if not result.get('success', False):
            print(f"错误信息: {result.get('error', 'N/A')}")
        
        
    except Exception as e:
        import traceback
        print(f"随机样本测试失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_test()