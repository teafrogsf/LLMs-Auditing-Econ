import json
import os
import re
import sys
import random
import networkx as nx
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))
from max_flow_solver import translate, evaluate
from max_flow_generator import Generator
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
        
    def load_random_graph(self, json_file_path: str = None):
        """
        从JSON文件中随机加载一个图
        
        Args:
            json_file_path: JSON文件路径
            
        Returns:
            G, (s,t)
        """
        if json_file_path is None:
            # 默认路径指向当前max_flow文件夹下的文件
            json_file_path = os.path.join(os.path.dirname(__file__), 'max_flow_graphs.json')
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 随机选择一个图
            graph_ids = list(data.keys())
            random_id = random.choice(graph_ids)
            graph_info = data[random_id]
            
            # 解析图数据
            graph_data = json.loads(graph_info['graph'])
            G = nx.node_link_graph(graph_data)
            
            # 获取源点和汇点
            if 'source' in graph_info and 'target' in graph_info:
                source, target = graph_info['source'], graph_info['target']
            else:
                raise ValueError("无法找到源点和汇点信息")
            
            q = (source, target)
            return G, q
            
        except Exception as e:
            print(f"从JSON文件加载图失败: {e}")
            return None
    
    def load_graph_by_index(self, graph_index: int, json_file_path: str = None):
        """
        从JSON文件中按索引加载指定的图
        
        Args:
            graph_index: 图的索引（0-999）
            json_file_path: JSON文件路径
            
        Returns:
            G, (s,t) 或 None（如果索引不存在）
        """
        if json_file_path is None:
            # 默认路径指向当前max_flow文件夹下的文件
            json_file_path = os.path.join(os.path.dirname(__file__), 'max_flow_graphs.json')
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 按索引选择图
            graph_key = str(graph_index)
            if graph_key not in data:
                print(f"图索引 {graph_index} 不存在")
                return None
                
            graph_info = data[graph_key]
            
            # 解析图数据
            graph_data = json.loads(graph_info['graph'])
            G = nx.node_link_graph(graph_data)
            
            # 获取源点和汇点
            if 'source' in graph_info and 'target' in graph_info:
                source, target = graph_info['source'], graph_info['target']
            else:
                raise ValueError("无法找到源点和汇点信息")
            
            q = (source, target)
            return G, q
            
        except Exception as e:
            print(f"从JSON文件加载图失败: {e}")
            return None
    
    def run_single_test(self, G, q):
        """
        运行单个测试
        
        Args:
            G: NetworkX图对象
            q: (source, target)
            
        Returns:
            测试结果
        """
        source, target = q
        # 计算正确答案
        try:
            correct_answer = nx.maximum_flow_value(G, source, target, capacity='capacity')
        except Exception as e:
            print(f"计算正确答案失败: {e}")
            return {
                'success': False,
                'error': f"计算正确答案失败: {e}"
            }
        
        # 创建提示
        pattern = "cot"
        prompt = translate(G, q, pattern)
        
        # 调用LLM
        # print(f"开始调用LLM")
        try:
            llm_answer, prompt_tokens, completion_tokens = self.llm.call_llm(prompt)
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
    # 创建运行器
    runner = MaxFlowRunner(model_name)
    # print(f"正在加载测试图...")
    # 从JSON文件中加载单个测试图
    graph_result = runner.load_random_graph()
    if graph_result is None:
        return {
            'success': False,
            'error': '从JSON文件加载图失败'
        }
    
    G, q = graph_result
    # print(f"测试图加载完成！")
    # 运行单个测试
    result = runner.run_single_test(G, q)
    return result


def run_multi_graph_test(model_name: str, num_tests: int = 10):
    """
    运行多个图测试
    
    Args:
        model_name: 模型名称
        num_tests: 测试数量，默认10个
        
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
    
    # 初始化锁
    file_lock = threading.Lock()
    stats_lock = threading.Lock()
    
    def run_single_test_wrapper(test_number):
        """单个测试的包装函数，用于并行执行"""
        print(f"\n=== 运行第 {test_number} 个测试样例 ===")
        
        # 从JSON文件中按顺序加载测试图（test_number从0开始）
        graph_result = runner.load_graph_by_index(test_number)
        
        if graph_result is None:
            print(f"测试 {test_number} 失败: 从JSON文件加载图失败")
            return {
                'success': False,
                'error': '从JSON文件加载图失败',
                'test_number': test_number
            }
        
        G, q = graph_result
        # 运行单个测试
        result = runner.run_single_test(G, q)
        result['test_number'] = test_number
        
        if result.get('success', False):
            score = result.get('score', 0)
            print(f"测试 {test_number} 完成，得分: {score}")
        else:
            score = 0
            print(f"测试 {test_number} 失败: {result.get('error', '未知错误')}")
    
        prompt_tokens = result.get('prompt_tokens', 0)
        completion_tokens = result.get('completion_tokens', 0)
        # 实时写入分数记录
        score_record_str = f"({test_number}:{score}:prompt={prompt_tokens}:completion={completion_tokens})"
        with file_lock:
            with open('max_flow_scores.txt', 'a', encoding='utf-8') as f:
                f.write(f"{model_name}:{score_record_str}\n")
        
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
    
    test_numbers = list(range(0, num_tests))
    
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
    
    # 所有测试完成后，添加分隔行
    with open('max_flow_scores.txt', 'a', encoding='utf-8') as f:
        f.write('\n\n\n')
    
    print(f"所有测试完成")
    
    summary = {
        'success': True,
        'total_tests': num_tests,
        'successful_tests': successful_tests,
        'average_score': average_score,
        'scores': scores,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens,
        'results': results
    }
    
    return summary


if __name__ == "__main__":
    run_single_graph_test("gpt-35-turbo")