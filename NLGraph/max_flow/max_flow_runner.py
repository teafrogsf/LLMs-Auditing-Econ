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
from llm_client import ExampleLLM

class MaxFlowRunner:
    def __init__(self, model_name: str = "deepseek-v3"):
        """
        初始化最大流测试运行器
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.test_data = self._load_test_data()
        self.llm = ExampleLLM(model_name)
        
    def _load_test_data(self) -> List[Dict]:
        """
        从main.json加载测试数据
        
        Returns:
            测试数据列表
        """
        json_path = os.path.join(os.path.dirname(__file__), "main.json")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"错误：找不到文件 {json_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"错误：解析JSON文件失败 - {e}")
            return []
    
    def _parse_graph_from_description(self, description: str) -> Tuple[nx.DiGraph, Tuple[int, int]]:
        """
        从问题描述中解析图结构
        
        Args:
            description: 问题描述文本
            
        Returns:
            (图对象, (源节点, 目标节点))
        """
        lines = description.strip().split('\n')
        
        # 解析节点数量
        first_line = lines[0]
        if "numbered from 0 to" in first_line:
            max_node = int(first_line.split("numbered from 0 to")[1].split(",")[0].strip())
            num_nodes = max_node + 1
        else:
            # 如果无法从描述中获取节点数，通过边来推断
            num_nodes = 0
            for line in lines[1:]:
                if "an edge from node" in line:
                    parts = line.split()
                    from_node = int(parts[4])
                    to_node = int(parts[7])
                    num_nodes = max(num_nodes, from_node + 1, to_node + 1)
        
        # 创建有向图
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        
        # 解析边和容量
        for line in lines[1:]:
            if "an edge from node" in line and "with capacity" in line:
                parts = line.split()
                from_node = int(parts[4])
                to_node = int(parts[7])
                capacity = int(parts[-1].rstrip('.,'))
                G.add_edge(from_node, to_node, capacity=capacity)
        
        # 解析源节点和目标节点
        question_line = [line for line in lines if line.startswith("Q:")][0]
        if "maximum flow from node" in question_line:
            parts = question_line.split("maximum flow from node")[1].split("to node")
            source = int(parts[0].strip())
            target = int(parts[1].strip().rstrip("?"))
        else:
            source, target = 0, num_nodes - 1  # 默认值
            
        return G, (source, target)
    

    

    
    def run_single_test(self, test_case: Dict) -> Dict[str, Any]:
        """
        运行单个测试用例
        
        Args:
            test_case: 测试用例数据
            
        Returns:
            测试结果
        """
        print(f"开始进行测试：")
        description = test_case.get('question', '')
        correct_answer_str = test_case.get('answer', '0')
        difficulty = test_case.get('difficulty', 'unknown')
        
        # 从答案字符串中提取数字（匹配最后一个数字）
        try:
            matches = re.findall(r'\d+', correct_answer_str)
            correct_answer = int(matches[-1]) if matches else 0

        except:
            correct_answer = 0
        
        # 解析图结构
        try:
            G, (source, target) = self._parse_graph_from_description(description)
        except Exception as e:
            print(f"解析图结构失败: {e}")
            return {
                'success': False,
                'correct': False,
                'error': f"图解析错误: {e}",
                'difficulty': difficulty
            }
        
        # 创建提示
        q = (source, target)
        args = type('Args', (), {
            'prompt': 'CoT'
        })()
        prompt = translate(G, q, args)
        
        # 调用LLM
        print(f"开始调用LLM")
        try:
            llm_answer, prompt_tokens, completion_tokens = self.llm.call_llm(prompt)
            print(f"LLM调用完成，开始评估答案...")
            
            # 评估答案
            q = (source, target)
            score = evaluate(llm_answer.lower(), G, q, correct_answer)

            # print(f"模型答案：{llm_answer}")
            # print(f"标准答案{correct_answer}")
            # print(f"最终评估：{score}")
            
            return {
                'success': True,
                'score': score,
                'llm_answer': llm_answer,
                'correct_answer': correct_answer,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'difficulty': difficulty,
                'source': source,
                'target': target
            }
            
        except Exception as e:
            return {
                'success': False,
                'correct': False,
                'error': str(e),
                'difficulty': difficulty
            }
    
    

def run_single_graph_test(model_name: str = "deepseek-v3"):
    """
    运行单个图测试
    
    Args:
        model_name: 模型名称
        
    Returns:
        测试结果
    """
    runner = MaxFlowRunner(model_name)
    
    # 随机选择一个测试用例
    if not runner.test_data:
        return {
            'success': False,
            'correct': False,
            'error': '没有可用的测试数据'
        }
    
    # 从所有可用的测试用例中随机选择一个
    available_indices = list(runner.test_data.keys())
    selected_test_id = random.choice(available_indices)
    test_case = runner.test_data[selected_test_id]
    
    print(f"随机选择的测试用例编号: {selected_test_id}")
    print(f"难度: {test_case.get('difficulty', 'unknown')}")
    
    # 运行单个测试
    result = runner.run_single_test(test_case)
    
    return result

def main():
    """
    主函数
    """
    model_name = "deepseek-v3"
    num_runs = 3
    
    print(f"最大流问题测试 - 模型: {model_name}")
    print(f"运行次数: {num_runs}")
    
    all_results = []
    total_correct = 0
    total_tests = 0
    
    for run in range(num_runs):
        print(f"\n{'='*50}")
        print(f"第 {run + 1} 次运行")
        print(f"{'='*50}")
        
        result = run_single_graph_test(model_name)
        all_results.append(result)
        
        if result['successful_tests'] > 0:
            total_correct += result['correct_answers']
            total_tests += result['successful_tests']
        
        print(f"\n第 {run + 1} 次运行结果:")
        print(f"生成结果：{result['correct']}")
        print(f"模型答案：{result['llm_answer']}, 标准答案：{result['correct_answer']}")
        print(f"输入tokens: {result['total_prompt_tokens']}, 输出tokens: {result['total_completion_tokens']}")
    
    # 计算总体统计
    if total_tests > 0:
        overall_accuracy = total_correct / total_tests
        
        print(f"\n{'='*50}")
        print(f"总体结果 ({num_runs} 次运行)")
        print(f"{'='*50}")
        print(f"总准确率: {overall_accuracy:.2%} ({total_correct}/{total_tests})")
   
        # 计算总token消耗
        total_prompt_tokens = sum(r.get('total_prompt_tokens', 0) for r in all_results)
        total_completion_tokens = sum(r.get('total_completion_tokens', 0) for r in all_results)
        print(f"总输入tokens: {total_prompt_tokens}, 总输出tokens: {total_completion_tokens}")
        
        # 计算每次运行的准确率
        accuracies = [r['accuracy'] for r in all_results if r['successful_tests'] > 0]
        if accuracies:
            print(f"准确率范围: {min(accuracies):.2%} - {max(accuracies):.2%}")
            print(f"准确率标准差: {(sum((a - overall_accuracy)**2 for a in accuracies) / len(accuracies))**0.5:.2%}")
    else:
        print("\n没有成功的测试结果")

if __name__ == "__main__":
    run_single_graph_test()