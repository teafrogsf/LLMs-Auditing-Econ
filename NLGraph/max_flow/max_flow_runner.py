import json
import os
import re
import sys
import random
import networkx as nx
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from flow import translate, evaluate
from flow_gen import Generator
from llm_client import ExampleLLM

class MaxFlowRunner:
    def __init__(self, model_name: str = "deepseek-v3"):
        """
        初始化最大流测试运行器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.llm = ExampleLLM(model_name)
        
    def generate_single_test_graph(self):
        """
        生成单个测试图
        
        Returns:
            tuple: (Graph, query)元组
        """
        
        n_min = 15
        n_max = 20
        num_nodes = random.randint(n_min, n_max)  # 随机节点数
        prob_min = 0.25
        prob_max = 0.3  # 边概率
        edge_prob = random.uniform(prob_min, prob_max)
        max_capacity = 20  # 最大容量
        
        generator = Generator(
            num_of_nodes=num_nodes,
            edge_probability=edge_prob,
            max_capacity=max_capacity
        )
        
        G, q = generator.generate()
        return G, q    

    
    def run_single_test(self, G, q):
        """
        运行单个测试
        
        Args:
            G: NetworkX图对象
            q: (source, target)
            
        Returns:
            测试结果
        """
        # print(f"开始进行测试：")
        source, target = q
        
        # 计算正确答案（使用NetworkX的最大流算法）
        try:
            correct_answer = nx.maximum_flow_value(G, source, target, capacity='capacity')
        except Exception as e:
            print(f"计算正确答案失败: {e}")
            return {
                'success': False,
                'error': f"计算正确答案失败: {e}"
            }
        
        # 创建提示
        args = type('Args', (), {'prompt': 'CoT'})()
        prompt = translate(G, q, args)
        
        # 调用LLM
        # print(f"开始调用LLM")
        try:
            llm_answer, prompt_tokens, completion_tokens = self.llm.call_llm(prompt)
            # print(f"LLM调用完成,开始评估答案...")
            
            # 评估答案
            score = evaluate(llm_answer.lower(), G, q, correct_answer)

            # print(f"模型答案：{llm_answer}")
            # print(f"标准答案：{correct_answer}")
            # print(f"最终评估：{score}")
            
            return {
                'success': True,
                'score': score,
                'llm_answer': llm_answer,
                'correct_answer': correct_answer,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'source': source,
                'target': target
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    

def run_single_graph_test(model_name: str):
    """
    运行单个图测试
    
    Args:
        model_name: 模型名称
        
    Returns:
        测试结果
    """
    # print(f"模型: {model_name}")
    
    # 创建运行器
    runner = MaxFlowRunner(model_name)
    
    # print(f"正在生成测试图...")
    # 生成单个测试图
    G, q = runner.generate_single_test_graph()
    
    # print(f"测试图生成完成！")
    # print(f"图信息: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
    # print(f"查询: 从节点{q[0]}到节点{q[1]}的最大流")
    # print(f"开始运行测试...")
    
    # 运行单个测试
    result = runner.run_single_test(G, q)
    
    return result


def run_multi_graph_test(model_name: str, num_tests: int = 100):
    """
    运行多个图测试
    
    Args:
        model_name: 模型名称
        num_tests: 测试数量，默认100个
        
    Returns:
        测试结果列表
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    runner = MaxFlowRunner(model_name)
    
    results = []
    total_score = 0
    successful_tests = 0
    scores = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    # 初始化分数记录列表和锁
    score_records = []
    file_lock = threading.Lock()
    stats_lock = threading.Lock()
    
    def run_single_test_wrapper(test_number):
        """单个测试的包装函数，用于并行执行"""
        print(f"\n=== 运行第 {test_number} 个测试样例 ===")
        
        # 生成测试图
        G, q = runner.generate_single_test_graph()
        
        print(f"测试 {test_number}: 图信息: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
        print(f"测试 {test_number}: 查询: 从节点{q[0]}到节点{q[1]}的最大流")
        
        # 运行单个测试
        result = runner.run_single_test(G, q)
        result['test_number'] = test_number
        
        if result.get('success', False):
            score = result.get('score', 0)
            print(f"测试 {test_number} 完成，得分: {score}")
        else:
            score = 0
            print(f"测试 {test_number} 失败: {result.get('error', '未知错误')}")
    
        score_record = f"({test_number}:{score})"
        with file_lock:
            with open('max_flow_scores.txt', 'a', encoding='utf-8') as f:
                f.write(f"{model_name}:{score_record}\n")
            score_records.append(score_record)
        
        # 更新统计信息
        with stats_lock:
            nonlocal total_score, successful_tests, total_prompt_tokens, total_completion_tokens
            results.append(result)
            if result.get('success', False):
                successful_tests += 1
                total_score += score
                scores.append(score)
            else:
                scores.append(0)
            
            total_prompt_tokens += result.get('prompt_tokens', 0)
            total_completion_tokens += result.get('completion_tokens', 0)
        
        return result
    
    test_numbers = list(range(1, num_tests + 1))
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 提交所有任务
        future_to_test = {executor.submit(run_single_test_wrapper, test_number): test_number for test_number in test_numbers}
        
        # 等待所有任务完成
        for future in as_completed(future_to_test):
            test_number = future_to_test[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'测试 {test_number} 产生异常: {exc}')
    
    # 计算统计信息
    average_score = total_score / num_tests if num_tests > 0 else 0
    max_score = max(scores) if scores else 0
    min_score = min(scores) if scores else 0
    
    # 所有测试完成后，添加分隔行
    with open('max_flow_scores.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n\n')
    
    print(f"所有测试完成")
    
    summary = {
        'success': True,
        'total_tests': num_tests,
        'successful_tests': successful_tests,
        'failed_tests': num_tests - successful_tests,
        'total_score': total_score,
        'average_score': average_score,
        'max_score': max_score,
        'min_score': min_score,
        'scores': scores,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'results': results
    }
    
    return summary


if __name__ == "__main__":
    run_single_graph_test("deepseek-v3")