import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加子目录到模块搜索路径
sys.path.append(os.path.join(current_dir, "max_flow"))
from max_flow_evaluator import run_single_graph_test
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

class MultiModelRunner:
    def __init__(self):
        """
        初始化多模型测试运行器
        """
        # 测试模型列表
        self.models = ['deepseek-v3','gpt-4o-mini','gpt-35-turbo','o1-mini']
        self.num_runs = 5  # 每个模型运行5次
        self.results = {}
    
    def run_single_model_test(self, model_name):
        """
        运行单个模型的测试
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 测试结果
        """
        print(f"\n开始测试模型: {model_name} (任务: max_flow)")
        print("-" * 50)
        
        scores = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for run_idx in range(self.num_runs):
            print(f"第 {run_idx + 1}/{self.num_runs} 次运行...")
            
            try:
                # 使用封装好的函数运行单次测试
                result = run_single_graph_test(model_name)
                
                scores.append(result['score'])
                total_prompt_tokens += result['prompt_tokens']
                total_completion_tokens += result['completion_tokens']
                
                print(f"  得分: {result['score']:.4f}")
                
            except Exception as e:
                print(f"  运行出错: {e}")
                scores.append(0)
        
        # 计算统计结果
        mean_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        
        result = {
            'model_name': model_name,
            'task_name': 'max_flow',
            'scores': scores,
            'mean_score': mean_score,
            'max_score': max_score,
            'min_score': min_score,
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens
        }
        
        return result
    
    def run_all_tests(self):
        """
        运行所有模型的测试
        """

        task_name = 'max_flow'
        # 初始化任务结果
        self.results[task_name] = {}
        
        # 对每个模型运行测试
        for model_name in self.models:
            try:
                result = self.run_single_model_test(model_name)
                self.results[task_name][model_name] = result
            except Exception as e:
                print(f"模型 {model_name} 运行出错: {e}")
                continue
    
    def plot_results(self):
        """
        绘制测试结果图表
        """
        if not self.results:
            print("没有测试结果可以绘制")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取结果数据
        task_name = 'max_flow'
        task_results = self.results[task_name]
        
        models = list(task_results.keys())
        mean_scores = [task_results[model]['mean_score'] for model in models]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制柱状图
        bars = ax.bar(models, mean_scores, alpha=0.7, color='#1f77b4')
        
        # 设置标题和标签
        ax.set_title('模型平均得分', fontsize=16, fontweight='bold')
        ax.set_ylabel('平均得分', fontsize=12)
        ax.set_xlabel('模型名称', fontsize=12)
        
        # 在柱子上显示数值
        for bar, mean_score in zip(bars, mean_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_score:.3f}', ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_model_results_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n结果图表已保存为: {filename}")
        
        plt.show()
    
    def save_results(self):
        """
        保存测试结果到JSON文件
        """
        if not self.results:
            print("没有测试结果可以保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_model_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        results_for_json = {}
        for task_name, task_results in self.results.items():
            results_for_json[task_name] = {}
            for model_name, model_result in task_results.items():
                results_for_json[task_name][model_name] = {
                    'model_name': model_result['model_name'],
                    'task_name': model_result['task_name'],
                    'scores': [float(s) for s in model_result['scores']],
                    'mean_score': float(model_result['mean_score']),
                    'max_score': float(model_result['max_score']),
                    'min_score': float(model_result['min_score']),
                    'total_prompt_tokens': model_result['total_prompt_tokens'],
                    'total_completion_tokens': model_result['total_completion_tokens']
                }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, ensure_ascii=False, indent=2)
        
        print(f"测试结果已保存为: {filename}")

def main():
    """
    主函数
    """
    runner = MultiModelRunner()
    
    try:
        # 运行所有测试
        runner.run_all_tests()
        
        # 保存结果
        runner.save_results()
        
        # 绘制图表
        runner.plot_results()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()