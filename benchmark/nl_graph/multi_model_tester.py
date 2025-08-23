import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_dir, "max_flow"))
from max_flow_evaluator import generate_and_evaluate_batch_graphs
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from loguru import logger
logger.add("logs/multi_model_tester.log", rotation="10 MB", retention="7 days", level="INFO")

class MultiModelRunner:
    def __init__(self, models=None, num_runs=1):
        """
        初始化多模型测试运行器
        
        Args:
            models: 测试模型列表
            num_runs: 测试次数，默认为 1
        """
        self.models = models
        self.num_runs = num_runs
        self.results = {}
    
    def save_single_model_result(self, model_name, result):
        """
        保存单个模型的测试结果到指定目录
        """
        save_dir = "e:/LLMs-Auditing-Econ/benchmark/nl_graph/max_flow/model_test_result/"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        filename = f"{model_name.replace('-', '_')}_test_result.jsonl"
        filepath = os.path.join(save_dir, filename)
        
        # 格式化分数列表，按图编号排序
        multi_test_summary = result.get('multi_test_summary', {})
        test_results = multi_test_summary.get('results', [])
        
        sorted_test_results = sorted(test_results, key=lambda x: x.get('test_number', 0))

        with open(filepath, 'w', encoding='utf-8') as f:
            for test_result in sorted_test_results:
                test_number = test_result.get('test_number', 0)
                score = test_result.get('score', 0)
                prompt_tokens = test_result.get('prompt_tokens', 0)
                completion_tokens = test_result.get('completion_tokens', 0)
                
                json_line = {
                    "ID": test_number,
                    "score": score,
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        logger.info(f"模型 {model_name} 的测试结果已保存到: {filepath}")
    
    def run_single_model_test(self, model_name):
        """
        运行单个模型的测试
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 测试结果
        """
        logger.info(f"\n开始测试模型: {model_name} (任务: max_flow)")
        logger.info("-" * 50)
        
        try:
            multi_result = generate_and_evaluate_batch_graphs(model_name, self.num_runs)
            
            if not multi_result.get('success', True):
                logger.error(f"  运行出错: {multi_result.get('error', '未知错误')}")
                result = {
                    'model_name': model_name,
                    'task_name': 'max_flow',
                    'scores': [0] * 5,
                    'mean_score': 0,
                    'total_prompt_tokens': 0,
                    'total_completion_tokens': 0
                }
                # 立即保存失败结果
                self.save_single_model_result(model_name, result)
                return result
            
            # 获取已计算的统计信息
            scores = multi_result.get('scores', [])
            mean_score = multi_result.get('average_score', 0)
            max_score = multi_result.get('max_score', 0)
            min_score = multi_result.get('min_score', 0)
            total_prompt_tokens = multi_result.get('total_prompt_tokens', 0)
            total_completion_tokens = multi_result.get('total_completion_tokens', 0)
            
            logger.info(f"  平均得分: {multi_result.get('average_score', 0):.4f}")
            logger.info(f"  成功测试数: {multi_result.get('successful_tests', 0)}/5")
            logger.info(f"  总Token使用量: {total_prompt_tokens + total_completion_tokens} (输入: {total_prompt_tokens}, 输出: {total_completion_tokens})")
            
            result = {
                'model_name': model_name,
                'task_name': 'max_flow',
                'scores': scores,
                'mean_score': mean_score,
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
                'multi_test_summary': multi_result  # 保存完整的多测试结果
            }
            
            # 立即保存成功结果
            self.save_single_model_result(model_name, result)
            return result
            
        except Exception as e:
            logger.error(f"  测试过程中发生异常: {str(e)}")
            result = {
                'model_name': model_name,
                'task_name': 'max_flow',
                'scores': [0] * 5,
                'mean_score': 0,
                'total_prompt_tokens': 0,
                'total_completion_tokens': 0
            }
            # 立即保存异常结果
            self.save_single_model_result(model_name, result)
            return result
    

    
    def plot_results(self):
        """
        绘制测试结果图表
        """
        if not self.results:
            logger.warning("没有测试结果可以绘制")
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取结果数据
        models = list(self.results.keys())
        mean_scores = [self.results[model]['mean_score'] for model in models]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制柱状图
        bars = ax.bar(models, mean_scores, alpha=0.7, color='#1f77b4')
        
        # 设置图表具体内容
        ax.set_title('模型平均得分', fontsize=16, fontweight='bold')
        ax.set_ylabel('平均得分', fontsize=12)
        ax.set_xlabel('模型名称', fontsize=12)
        for bar, mean_score in zip(bars, mean_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_score:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_model_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"\n结果图表已保存为: {filename}")
        
        plt.show()
    
    

def main():
    test_models = ['deepseek-v3']
    test_runs = 2
    
    logger.info(f"开始测试，模型列表: {test_models}，测试次数: {test_runs}")
    
    runner = MultiModelRunner(models=test_models, num_runs=test_runs)
    
    try:
        # 运行所有模型的测试
        for model_name in runner.models:
            try:
                result = runner.run_single_model_test(model_name)
                runner.results[model_name] = result
            except Exception as e:
                logger.error(f"模型 {model_name} 运行出错: {e}")
                continue
        
        runner.plot_results()
        
    except KeyboardInterrupt:
        logger.warning("\n测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()