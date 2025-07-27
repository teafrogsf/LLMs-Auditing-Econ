import os
import json
import re
import random
from typing import Dict, List, Tuple, Any
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# 导入现有模块
from latex_pre_process import master_convert
from EED import EED
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from llm_client import ExampleLLM

class PHYBenchEvaluator:
    def __init__(self, models_to_test: List[str] = None):
        """
        初始化PHYBench评测器
        
        Args:
            models_to_test: 要测试的模型列表，如果为None则测试所有可用模型
        """
        self.prompt_template = (
            "Please read the following question and provide a step-by-step solution. "
            "Put your final answer, which must be a readable LaTeX formula, in a \\boxed{{}} environment.\n\n"
            "Question: {problem}\n\n"
            "Answer:"
        )
        
        # 可用的模型列表
        self.available_models = [
            "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-35-turbo-0125-60ktpm",
            "o1-mini-1mtpm", "o1", "o3-mini-1mtpm",
            "qwen-max", "deepseek-r1", "deepseek-v3", "deepseek-chat", "deepseek-reasoner"
        ]
        
        if models_to_test is None:
            self.models_to_test = self.available_models
        else:
            self.models_to_test = [m for m in models_to_test if m in self.available_models]
            
        # print(f"将测试以下模型: {self.models_to_test}")
        
    def load_dataset(self, file_path: str = None, num_samples: int = None) -> List[Dict]:
        """
        加载PHYBench数据集
        
        Args:
            file_path: JSON文件路径，如果为None则尝试加载PHYBench-fullques.json
            num_samples: 限制样本数量，如果为None则使用全部数据
            
        Returns:
            数据集样本列表
        """
        if file_path is None:
            # print("正在加载PHYBench-fullques.json (100个完整解答样本)...")
            try:
                from huggingface_hub import hf_hub_download
                file_path = hf_hub_download(
                    repo_id="Eureka-Lab/PHYBench",
                    filename="PHYBench-fullques_v1.json",
                    repo_type="dataset"
                )
            except Exception as e:
                print(f"从Hugging Face下载失败: {e}")
                print("请手动下载PHYBench-fullques.json文件")
                return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 确保数据是列表格式
            if isinstance(data, dict):
                data = list(data.values())
            
            if num_samples is not None:
                data = data[:num_samples]
                
            # print(f"成功加载 {len(data)} 个样本")
            return data
        except Exception as e:
            print(f"加载数据集失败: {e}")
            return []
    
    def extract_boxed_answer(self, response: str) -> str:
        """
        从模型响应中提取\boxed{}中的答案
        
        Args:
            response: 模型的完整响应
            
        Returns:
            提取的LaTeX答案，如果没找到则返回空字符串
        """
        # 查找\boxed{的起始位置
        start_pattern = r'\\boxed\{'
        matches = list(re.finditer(start_pattern, response))
        
        if not matches:
            return ""
        
        # 对每个\boxed{进行花括号平衡
        extracted_answers = []
        for match in matches:
            start_pos = match.end() - 1  # 指向开始的{
            brace_count = 0
            i = start_pos
            
            while i < len(response):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # 找到匹配的}
                        content = response[start_pos + 1:i]
                        extracted_answers.append(content)
                        # print(f"提取到的boxed内容: '{content}'")
                        break
                i += 1
        
        if extracted_answers:
            return extracted_answers[-1]  # 返回最后一个匹配的答案
                
        return ""
    
    def extract_ground_truth_answer(self, answer: str) -> str:
        """
        从数据集答案中提取LaTeX表达式
        
        Args:
            answer: 数据集中的原始答案
            
        Returns:
            提取的LaTeX答案，如果没找到则返回空字符串
        """
        # 先查找$$...$$格式
        pattern_dollar = r'\$\$([^$]+)\$\$'
        matches = re.findall(pattern_dollar, answer)
        
        if matches:
            return matches[-1].strip()  # 返回最后一个匹配的答案并去除首尾空格
        
        # 如果没找到$$格式，查找\[...\]格式
        pattern_bracket = r'\\\[([^\]]+)\\\]'
        matches = re.findall(pattern_bracket, answer)
        
        if matches:
            return matches[-1].strip()  # 返回最后一个匹配的答案并去除首尾空格
                
        return ""
    
    def normalize_latex_brackets(self, latex_str: str) -> str:
        """
        统一处理LaTeX中的\\left和\\right括号，将其替换为普通括号
        """
        if not latex_str:
            return latex_str
        
        # 替换\left和\right括号
        latex_str = latex_str.replace('\\left(', '(')
        latex_str = latex_str.replace('\\right)', ')')
        latex_str = latex_str.replace('\\left[', '[')
        latex_str = latex_str.replace('\\right]', ']')
        latex_str = latex_str.replace('\\left\\{', '{')
        latex_str = latex_str.replace('\\right\\}', '}')
        latex_str = latex_str.replace('\\left|', '|')
        latex_str = latex_str.replace('\\right|', '|')
        
        return latex_str
    
    def evaluate_single_sample(self, model: ExampleLLM, sample: Dict, _retry_count: int = 0) -> Dict:
        """
        评估单个样本
        
        Args:
            model: LLM模型实例
            sample: 数据集样本
            _retry_count: 内部重试计数器（私有参数）
            
        Returns:
            评估结果字典
        """
        try:
            # 构建prompt - 使用正确的字段名
            problem_text = sample.get('content', '')
            if not problem_text:
                raise KeyError(f"未找到问题文本字段，可用字段: {list(sample.keys())}")
            prompt = self.prompt_template.format(problem=problem_text)
            
            # 调用模型
            response, prompt_tokens, completion_tokens = model.call_llm(prompt)
            # print(f"模型响应长度: {len(response) if response else 0}")
            
            # 提取答案
            predicted_answer = self.extract_boxed_answer(response)
            # print(f"提取的预测答案: '{predicted_answer}'")
            
            # 获取标准答案并提取其中的latex
            raw_ground_truth = sample.get('answer', '')
            # print(f"原始标准答案: '{raw_ground_truth}'")
            ground_truth = self.extract_ground_truth_answer(raw_ground_truth) if raw_ground_truth else ''
            # print(f"提取的标准答案: '{ground_truth}'")
            
            
            # 计算EED
            # print(f"开始计算EED分数（使用原始LaTeX字符串）...")
            if predicted_answer and ground_truth:
                try:
                    # 防御性转换
                    pred_latex_eed = str(predicted_answer)
                    truth_latex_eed = str(ground_truth)
                    
                    # 统一处理\left和\right括号
                    pred_latex_eed = self.normalize_latex_brackets(pred_latex_eed)
                    truth_latex_eed = self.normalize_latex_brackets(truth_latex_eed)
                    
                    print(f"EED格式预测答案: {pred_latex_eed}")
                    print(f"EED格式标准答案: {truth_latex_eed}")
                    
                    # 保存传给EED的字符串用于调试
                    eed_predicted_input = pred_latex_eed
                    eed_ground_truth_input = truth_latex_eed
                    
                    eed_score, relative_distance, answer_tree_size, distance = EED(truth_latex_eed, pred_latex_eed)
                    # print(f"EED计算成功: score={eed_score}, rel_dist={relative_distance}, tree_size={answer_tree_size}, dist={distance}")
                except Exception as e:
                    print(f"EED计算错误: {e}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    eed_score, relative_distance, answer_tree_size, distance = 0, -1, -1, -1
            else:
                print(f"预测答案或标准答案为空，无法计算EED")
                eed_score, relative_distance, answer_tree_size, distance = 0, -1, -1, -1
            
            return {
                'problem_id': sample.get('id', ''),
                'problem': problem_text,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'full_response': response,
                'eed_score': eed_score,
                'relative_distance': relative_distance,
                'answer_tree_size': answer_tree_size,
                'distance': distance,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'eed_predicted_input': eed_predicted_input,
                'eed_ground_truth_input': eed_ground_truth_input,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            import traceback
            print(f"评估样本时出错: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            
            # 如果还有重试次数，则重新调用函数进行重试（最多3次）
            if _retry_count < 3:
                print(f"评估失败，还有 {3 - _retry_count} 次重试机会，正在重新评估...")
                return self.evaluate_single_sample(model, sample, _retry_count + 1)
            
            # 重试次数用完，返回失败结果
            print(f"评估失败，重试次数已用完")
            return {
                'problem_id': sample.get('id', ''),
                'problem': problem_text if 'problem_text' in locals() else str(sample),
                'ground_truth': sample.get('answer', ''),
                'predicted_answer': '',
                'full_response': '',
                'eed_score': 0,
                'relative_distance': -1,
                'answer_tree_size': -1,
                'distance': -1,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'success': False,
                'error': str(e)
            }
    
    
    def test_random_sample(self, model_name: str, dataset: List[Dict] = None) -> Tuple[float, int, int]:
        """
        随机选择数据集中的一个样本，测试指定模型并返回EED分数和token信息
        
        Args:
            model_name: 要测试的模型名称
            dataset: 数据集，如果为None则自动加载完整数据集
            
        Returns:
            Tuple[float, int, int]: (EED分数, 输入token数, 输出token数)，如果评估失败返回(0.0, 0, 0)
        """
        try:
            # 如果没有提供数据集，则加载完整数据集
            if dataset is None:
                # print("正在加载数据集...")
                dataset = self.load_dataset(None, None)  # 加载所有样本
                if not dataset:
                    print("数据集加载失败")
                    return 0.0, 0, 0
            
            # 检查数据集是否为空
            if len(dataset) == 0:
                print("数据集为空")
                return 0.0, 0, 0
            
            # 随机选择一个样本
            selected_sample = random.choice(dataset)
            # print(f"随机选择了样本ID: {selected_sample.get('id', 'N/A')}")
            
            # 初始化模型
            # print(f"正在初始化模型: {model_name}")
            model = ExampleLLM(model_name)
            
            # 评估单个样本
            # print("正在评估样本...")
            result = self.evaluate_single_sample(model, selected_sample)
            
            # 返回EED分数和token信息
            eed_score = result.get('eed_score', 0.0)
            prompt_tokens = result.get('prompt_tokens', 0)
            completion_tokens = result.get('completion_tokens', 0)
            # print(f"评估完成，EED分数: {eed_score}, 输入tokens: {prompt_tokens}, 输出tokens: {completion_tokens}")
            
            # 可选：打印更多详细信息
            if result.get('success', False) == False:
                print(f"评估失败: {result.get('error', '未知错误')}")
            else:
                relative_distance = result.get('relative_distance', -1)
                print(f"评估成功，相对距离: {relative_distance}")
            
            return float(eed_score), int(prompt_tokens), int(completion_tokens)
            
        except Exception as e:
            import traceback
            print(f"随机样本测试失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            return 0.0, 0, 0


def main():
    """
    主函数 - 运行评估
    """
    # 要测试的模型
    models_to_test = [
        "gpt-4o-mini",
        "gpt-4o", 
        "qwen-max",
        # "deepseek-v3", 
    ]
    
    evaluator = PHYBenchEvaluator(models_to_test=models_to_test)
    
    # 运行评估
    results = evaluator.run_evaluation(
        file_path=None,        # 自动下载PHYBench-fullques.json，或指定本地文件路径
        num_samples=50,        # 限制为50个样本进行快速测试，设为None使用全部100个样本
        save_results=True      # 保存结果
    )
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()