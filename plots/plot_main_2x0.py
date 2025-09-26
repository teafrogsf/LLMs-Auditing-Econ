import json
import os
import itertools
from random import choice
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

CHOICES = [str(item) for item in list(range(10))]
# RESULTS_PATH = Path('outputs/default')
# CONFIG_PATH = Path('config/default.yaml')

def plot_histogram(provider1_data, provider2_results, save_path, choices=CHOICES):
    os.makedirs(save_path, exist_ok=True)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')

    cmap = plt.get_cmap('tab10')
    strategy_colors = [cmap(i % 10) for i in range(len(choices))]
    provider1_color = 'green'  # 服务商1固定颜色

    metrics = [
        ('provider_utility', 'Provider Utility'),
        ('user_utility', 'User Utility'),
        ('delegations', 'Delegations'),
    ]

    # Create 1x3 subplot grid (合并显示)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for j, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[j]
        
        # 计算柱子位置：服务商1的柱子在左侧，服务商2的柱子在右侧
        num_strategies = len(choices)
        bar_width = 0.35
        
        # 服务商1的数据（每个策略对应的所有值）
        provider1_vals = provider1_data[metric_key]
        
        # 服务商2的数据（每个策略的平均值）
        provider2_vals = [provider2_results[strategy][metric_key] for strategy in choices]
        
        # x轴位置
        x_positions = np.arange(num_strategies)
        
        # 绘制服务商1的柱子（左侧）
        bars1 = ax.bar(x_positions - bar_width/2, provider1_vals, bar_width, 
                      color=provider1_color, edgecolor='black', linewidth=0.6, 
                      label='Provider 1 (Strategy 3)', alpha=0.8)
        
        # 绘制服务商2的柱子（右侧）
        bars2 = ax.bar(x_positions + bar_width/2, provider2_vals, bar_width,
                      color=strategy_colors, edgecolor='black', linewidth=0.6,
                      label='Provider 2', alpha=0.8)
        
        ax.set_title(f"{metric_label} Comparison", fontsize=14, pad=15)
        ax.set_ylabel(metric_label, fontsize=12)
        
        # 不显示x轴标签，仅通过颜色区分策略
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=11)
        ax.margins(y=0.15)
        
        # 添加数值标注
        for rect, val in zip(bars1, provider1_vals):
            height = rect.get_height()
            ax.annotate(f"{val:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        for rect, val in zip(bars2, provider2_vals):
            height = rect.get_height()
            ax.annotate(f"{val:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    fig.suptitle("Provider Performance Comparison (3-x-0 Strategy)", fontsize=16, y=0.95)

    # 创建图例
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=provider1_color, edgecolor='black', label='Provider 1 (Strategy 3)')]
    legend_handles.extend([Patch(facecolor=strategy_colors[i], edgecolor='black', 
                                label=f'Provider 2 (Strategy {int(choices[i]) + 1})')
                          for i in range(len(choices))])
    
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(6, len(legend_handles)), 
               frameon=False, bbox_to_anchor=(0.5, 0.02), fontsize=10)

    fig.tight_layout(rect=[0.02, 0.12, 1, 0.92])

    outfile = save_path / "provider_comparison_3x0.pdf"
    fig.savefig(outfile, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图表已保存到: {outfile}")

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--config', type=str, default='config/nl_graph/default.yaml')
    paser.add_argument('--output-dir', type=str, default='outputs/nl_graph/v5')
    args = paser.parse_args()
    CONFIG_PATH = Path(args.config)
    RESULTS_PATH = Path(args.output_dir)
    scenarios = [f'3-{item}-0' for item in CHOICES]
    config = yaml.safe_load(open(CONFIG_PATH))



    num_providers = len(config['providers'])
    num_others = num_providers - 1


    # 收集Provider 1的数据 (索引为0，固定策略3) - 按策略分组
    provider1_data = {
        'provider_utility': [],
        'user_utility': [],
        'delegations': []
    }
    
    # 收集Provider 2的数据 (索引为1，策略变化)
    provider2_cost = {k: [] for k in CHOICES}
    provider2_reward = {k: [] for k in CHOICES}
    provider2_price = {k: [] for k in CHOICES}
    provider2_utility = {k: [] for k in CHOICES} 
    provider2_user_utility = {k: [] for k in CHOICES}
    provider2_results = {k: {} for k in CHOICES}
    provider2_delegations = {k: [] for k in CHOICES}
    
    for strategy in CHOICES:
        all_scenarios = [f'2-{strategy}-0']
        print(all_scenarios)
        results = [json.load(open(RESULTS_PATH / item / 'result.json')) for item in all_scenarios]
        
        for result in results:
            # Provider 1 (索引0) - 固定策略3，收集每次的值
            provider1_data['provider_utility'].append(result['providers'][0]['total_provider_utility'])
            provider1_data['user_utility'].append(result['providers'][0]['total_user_utility'])
            provider1_data['delegations'].append(result['providers'][0]['delegations'])
            
            # Provider 2 (索引1) - 策略变化
            provider2_cost[strategy].append(result['providers'][1]['total_cost']) 
            provider2_price[strategy].append(result['providers'][1]['total_price']) 
            provider2_reward[strategy].append(result['providers'][1]['total_reward']) 
            provider2_utility[strategy].append(result['providers'][1]['total_provider_utility']) 
            provider2_user_utility[strategy].append(result['providers'][1]['total_user_utility']) 
            provider2_delegations[strategy].append(result['providers'][1]['delegations'])
            
        provider2_results[strategy]['avg_cost'] = np.mean(provider2_cost[strategy])
        provider2_results[strategy]['avg_price'] = np.mean(provider2_price[strategy])
        provider2_results[strategy]['avg_reward'] = np.mean(provider2_reward[strategy])
        provider2_results[strategy]['provider_utility'] = np.mean(provider2_utility[strategy])
        provider2_results[strategy]['user_utility'] = np.mean(provider2_user_utility[strategy])
        provider2_results[strategy]['delegations'] = np.mean(provider2_delegations[strategy])
    
    plot_histogram(provider1_data, provider2_results, save_path=RESULTS_PATH / 'figs')