#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    evaluator = PHYBenchEvaluator(models_to_test=["deepseek-chat"])
    
    # 加载数据集
    print("加载数据集...")
    dataset = evaluator.load_dataset(None, None)  # 加载所有样本
    
    if not dataset:
        print("数据集加载失败")
        return
    
    print(f"成功加载 {len(dataset)} 个样本")
    
    # 随机选择一个样本
    selected_sample = random.choice(dataset)
    print(f"随机选择的样本字段: {list(selected_sample.keys())}")
    print(f"随机选择的样本ID: {selected_sample.get('id', 'N/A')}")
    print(f"随机选择的样本标签: {selected_sample.get('tag', 'N/A')}")
    print(f"随机选择的样本答案: {selected_sample.get('answer', 'N/A')[:200]}...")  # 只显示前200个字符
    
    # 测试模型初始化
    print("\n测试模型初始化...")
    try:
        from llm_client import ExampleLLM
        model = ExampleLLM("deepseek-v3")
        print("模型初始化成功")
        
        # 测试单个样本评估
        print("\n测试单个样本评估...")
        result = evaluator.evaluate_single_sample(model, selected_sample)
        
        print("\n=== 详细评估结果 ===")
        print(f"问题ID: {result.get('problem_id', 'N/A')}")
        print(f"\n完整模型回答:")
        print(f"{result.get('full_response', 'N/A')}")
        
        # 打印传给EED的两个字符串
        if 'eed_predicted_input' in result:
            print(f"\n传给EED的预测答案字符串: '{result.get('eed_predicted_input', '')}'")
        if 'eed_ground_truth_input' in result:
            print(f"传给EED的标准答案字符串: '{result.get('eed_ground_truth_input', '')}'")
        
        print(f"\nEED分数: {result.get('eed_score', 0)}")
        print(f"相对距离: {result.get('relative_distance', -1)}")
        print(f"答案树大小: {result.get('answer_tree_size', -1)}")
        print(f"距离: {result.get('distance', -1)}")
        print(f"\nToken使用情况:")
        print(f"输入tokens: {result.get('prompt_tokens', 0)}")
        print(f"输出tokens: {result.get('completion_tokens', 0)}")
        print(f"\n评估成功: {result.get('success', False)}")
        if result.get('error'):
            print(f"错误信息: {result.get('error')}")
        
    except Exception as e:
        import traceback
        print(f"模型初始化失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_test()