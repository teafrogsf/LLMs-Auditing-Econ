import json
import os
import re
import sys
import random
import networkx as nx
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from matching import translate,evaluate
from llm_client import ExampleLLM

class BipartiteMatchRunner:
    def __init__(self, model_name: str = "deepseek-v3"):
        """
        初始化二分匹配测试运行器
        
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
    
    def _parse_graph_from_description(self, description: str) -> Tuple[nx.Graph, Tuple[int, int]]:
        """
        从问题描述中解析二分匹配图结构
        
        Args:
            description: 问题描述文本
            
        Returns:
            (二分图对象, (申请者数量, 工作数量))
        """
        lines = description.strip().split('\n')
        
        # 解析申请者和工作数量
        first_line = lines[0]
        # 提取申请者数量
        if "job applicants numbered from 0 to" in first_line:
            applicant_max = int(first_line.split("job applicants numbered from 0 to")[1].split(",")[0].strip())
            n1 = applicant_max + 1
        else:
            n1 = 0
            
        # 提取工作数量
        if "jobs numbered from 0 to" in first_line:
            job_max = int(first_line.split("jobs numbered from 0 to")[1].split(".")[0].strip())
            n2 = job_max + 1
        else:
            n2 = 0
        
        # 创建二分图
        G = nx.Graph()
        # 添加申请者节点 (0 到 n1-1)
        G.add_nodes_from(range(n1), bipartite=0)
        # 添加工作节点 (n1 到 n1+n2-1)
        G.add_nodes_from(range(n1, n1 + n2), bipartite=1)
        
        # 解析申请者对工作的兴趣（边）
        for line in lines[1:]:
            if "Applicant" in line and "is interested in job" in line:
                # 解析申请者编号
                applicant_part = line.split("Applicant")[1].split("is interested in job")
                applicant_id = int(applicant_part[0].strip())
                # 解析工作编号
                job_id = int(applicant_part[1].strip().rstrip('.,'))
                # 添加边（申请者节点ID，工作节点ID = n1 + job_id）
                G.add_edge(applicant_id, n1 + job_id)
            
        return G, (n1, n2)
    
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
        
        # 提取答案
        try:
            matches = re.findall(r'\d+', correct_answer_str)
            correct_answer = int(matches[-1]) if matches else 0

        except:
            correct_answer = 0
        
        # 解析图结构
        try:
            G, (n1, n2) = self._parse_graph_from_description(description)
        except Exception as e:
            print(f"解析图结构失败: {e}")
            return {
                'success': False,
                'correct': False,
                'error': f"图解析错误: {e}",
                'difficulty': difficulty
            }
        
        # 创建提示
        args = type('Args', (), {'prompt': 'CoT'})()
        prompt = translate(G, n1, n2 , args)
        
        # 调用LLM
        print(f"开始调用LLM")
        try:
            llm_answer, prompt_tokens, completion_tokens = self.llm.call_llm(prompt)
            print(f"LLM调用完成，开始评估答案...")
            
            # 评估答案
            score = evaluate(llm_answer.lower(), G, n1, correct_answer)

            print(f"模型答案：{llm_answer}")
            print(f"标准答案{correct_answer}")
            print(f"最终评估：{score}")
            
            return {
                'success': True,
                'score': score,
                'llm_answer': llm_answer,
                'correct_answer': correct_answer,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'difficulty': difficulty,
                'n1': n1,
                'n2': n2
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
    runner = BipartiteMatchRunner(model_name)
    
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

if __name__ == "__main__":
    run_single_graph_test()