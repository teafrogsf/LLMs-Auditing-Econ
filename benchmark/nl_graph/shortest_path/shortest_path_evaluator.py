import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shortest_path_generator import Generator
from shortest_path_solver import translate, evaluate
from llm_client import ExampleLLM
import networkx as nx
import argparse
import random
import numpy as np
from datetime import datetime, timedelta, timezone

class ShortestPathRunner:
    def __init__(self, model_name="deepseek-v3", prompt="none", city=0):
        """
        初始化最短路径测试运行器
        
        Args:
            model_name: 使用的模型名称")
            prompt: 提示技术
            city: 是否使用城市描述 (0或1)
        """
        self.model_name = model_name
        self.prompt = prompt
        self.city = city
        self.llm = ExampleLLM(model_name)
        
        # 模拟args对象用于兼容现有函数
        self.args = type('Args', (), {
            'prompt': prompt,
            'city': city,
            'model': model_name
        })()
    
    def generate_single_test_graph(self):
        """
        生成单个测试图
        
        Returns:
            tuple: (Graph, query)元组
        """
        
        n_min = 20
        n_max = 30
        num_nodes = random.randint(n_min, n_max)  # 固定节点数
        edge_prob = 0.3  # 固定边概率
        max_weight = 15
        
        generator = Generator(
            num_of_nodes=num_nodes,
            edge_probability=edge_prob,
            max_weight=max_weight
        )
        
        G, q = generator.generate()
        return G, q
    
    def run_single_test(self, G, q):
        """
        运行单个测试
        
        Args:
            G: NetworkX图对象
            q: 查询 [start_node, end_node]
            
        Returns:
            tuple: (score, answer, prompt)
        """
        print(f"正在生成提示...")
        # 生成提示
        prompt = translate(G, q, self.args)
        print(f"提示生成完成，开始调用LLM...")
        
        # 调用LLM
        try:
            answer, prompt_tokens, completion_tokens = self.llm.call_llm(prompt)
            print(f"LLM调用完成，开始评估答案...")
            
            # 评估答案
            score = evaluate(answer.lower(), G, q)
            print(f"评估完成，得分: {score:.3f}")
            
            return score, answer, prompt, prompt_tokens, completion_tokens
        except Exception as e:
            print(f"Error in LLM call: {e}")
            return 0, "", prompt, 0, 0
    
    
    
    def print_results(self, results):
        """
        打印测试结果
        
        Args:
            results: 测试结果字典
        """
        print("\n" + "="*50)
        print("测试结果统计")
        print("="*50)
        print(f"模型: {results['model_name']}")
        print(f"提示技术: {results['prompt']}")
        print(f"测试图数量: {results['num_graphs']}")
        print(f"平均得分: {results['mean_score']:.4f}")
        print(f"最高得分: {results['max_score']:.4f}")
        print(f"最低得分: {results['min_score']:.4f}")
        print(f"总输入tokens: {results['total_prompt_tokens']}")
        print(f"总输出tokens: {results['total_completion_tokens']}")
        print("="*50)

def run_single_graph_test(model: str):
    """
    运行单个图的测试
    """
    model_name = model
    prompt = "CoT"  # 使用CoT提示
    city = 0
    
    print(f"初始化测试运行器...")
    print(f"模型: {model_name}, 提示技术: {prompt}")
    
    # 创建运行器
    runner = ShortestPathRunner(
        model_name=model_name,
        prompt=prompt,
        city=city
    )
    
    print(f"正在生成测试图...")
    # 生成单个测试图
    G, q = runner.generate_single_test_graph()
    
    print(f"测试图生成完成！")
    print(f"图信息: {G.number_of_nodes()}个节点, {G.number_of_edges()}条边")
    print(f"查询: 从节点{q[0]}到节点{q[1]}的最短路径")
    print(f"开始运行测试...")
    
    # 运行单个测试
    score, answer, prompt, prompt_tokens, completion_tokens = runner.run_single_test(G, q)
    
    # 打印详细结果
    print("\n" + "="*50)
    print("测试结果")
    print("="*50)
    print(f"模型: {model_name}")
    print(f"得分: {score:.4f}")
    print(f"输入tokens: {prompt_tokens}")
    print(f"输出tokens: {completion_tokens}")
    print("="*50)
    
    return {
        'score': score,
        'answer': answer,
        'prompt': prompt,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'graph': G,
        'query': q
    }

def main():
    """
    运行10次单次测试并统计结果
    """
    print("开始运行10次单次测试...")
    
    all_scores = []
    all_prompt_tokens = []
    all_completion_tokens = []
    
    for i in range(3):
        print(f"\n第{i+1}次测试:")
        print("-" * 30)
        
        result = run_single_graph_test("deepseek-v3")
        
        all_scores.append(result['score'])
        all_prompt_tokens.append(result['prompt_tokens'])
        all_completion_tokens.append(result['completion_tokens'])
    
    # 计算统计结果
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    max_score = np.max(all_scores)
    min_score = np.min(all_scores)
    total_prompt_tokens = sum(all_prompt_tokens)
    total_completion_tokens = sum(all_completion_tokens)
    
    # 打印最终统计结果
    print("\n" + "="*60)
    print("10次测试总体统计结果")
    print("="*60)
    print(f"平均得分: {mean_score:.4f}")
    print(f"最高得分: {max_score:.4f}")
    print(f"最低得分: {min_score:.4f}")
    print(f"总输入tokens: {total_prompt_tokens}")
    print(f"总输出tokens: {total_completion_tokens}")
    print(f"各次得分: {[f'{score:.3f}' for score in all_scores]}")
    print("="*60)
    
    return {
        'all_scores': all_scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'max_score': max_score,
        'min_score': min_score,
        'total_prompt_tokens': total_prompt_tokens,
        'total_completion_tokens': total_completion_tokens
    }

if __name__ == "__main__":
    main()