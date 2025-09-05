import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_log_file(log_file_path):
    """解析单个日志文件，提取关键指标"""
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取策略组合
        filename = os.path.basename(log_file_path)
        strategy_match = re.search(r'simulator_(.+)\.log', filename)
        if not strategy_match:
            return None
        strategies = strategy_match.group(1).split('-')
        
        # 提取博弈结果
        results = {}
        
        # 提取总时间步数和实际委托次数
        total_time_match = re.search(r'总时间步数：(\d+)', content)
        total_delegations_match = re.search(r'实际委托次数：(\d+)', content)
        best_provider_match = re.search(r'最佳服务商：(\d+)', content)
        
        if total_time_match:
            results['total_time'] = int(total_time_match.group(1))
        if total_delegations_match:
            results['total_delegations'] = int(total_delegations_match.group(1))
        if best_provider_match:
            results['best_provider'] = int(best_provider_match.group(1))
        
        # 提取各服务商统计信息
        provider_stats = {}
        provider_pattern = r'服务商(\d+):\s*\n.*?委托次数：(\d+)\s*\n.*?总价格：([\d.]+)\s*\n.*?总成本：([\d.]+)\s*\n.*?服务商效用：([\d.]+)\s*\n.*?总回报：([\d.]+)\s*\n.*?平均回报：([\d.]+)\s*\n.*?用户效用：([\d.]+)'
        
        for match in re.finditer(provider_pattern, content, re.DOTALL):
            provider_id = int(match.group(1))
            provider_stats[provider_id] = {
                'delegations': int(match.group(2)),
                'total_price': float(match.group(3)),
                'total_cost': float(match.group(4)),
                'provider_utility': float(match.group(5)),
                'total_reward': float(match.group(6)),
                'avg_reward': float(match.group(7)),
                'user_utility': float(match.group(8))
            }
        
        results['provider_stats'] = provider_stats
        results['strategies'] = strategies
        
        return results
    except Exception as e:
        print(f"解析文件 {log_file_path} 时出错: {e}")
        return None

def load_all_results():
    """加载所有日志文件的结果"""
    logs_dir = Path('logs')
    all_results = []
    
    for log_file in logs_dir.glob('simulator_*.log'):
        result = parse_log_file(log_file)
        if result:
            all_results.append(result)
    
    return all_results

def create_summary_dataframe(all_results):
    """创建汇总数据框"""
    summary_data = []
    
    for result in all_results:
        strategies = result['strategies']
        total_user_utility = sum(stats['user_utility'] for stats in result['provider_stats'].values())
        total_provider_utility = sum(stats['provider_utility'] for stats in result['provider_stats'].values())
        total_reward = sum(stats['total_reward'] for stats in result['provider_stats'].values())
        
        # 计算各策略的效用
        strategy_utilities = {}
        for i, strategy in enumerate(strategies, 1):
            if i in result['provider_stats']:
                strategy_utilities[f'provider_{i}_{strategy}'] = result['provider_stats'][i]['provider_utility']
        
        summary_data.append({
            'strategy_combination': '-'.join(strategies),
            'provider1_strategy': strategies[0],
            'provider2_strategy': strategies[1], 
            'provider3_strategy': strategies[2],
            'total_user_utility': total_user_utility,
            'total_provider_utility': total_provider_utility,
            'total_reward': total_reward,
            'total_delegations': result['total_delegations'],
            'best_provider': result['best_provider'],
            **strategy_utilities
        })
    
    return pd.DataFrame(summary_data)

def create_visualizations(df):
    """创建可视化图表"""
    
    # 1. 不同策略组合的总效用对比
    plt.figure(figsize=(15, 10))
    
    # 子图1: 用户效用对比
    plt.subplot(2, 3, 1)
    strategy_means = df.groupby('provider1_strategy')['total_user_utility'].mean()
    strategy_means.plot(kind='bar', color='skyblue')
    plt.title('不同策略下的平均用户效用')
    plt.ylabel('用户效用')
    plt.xticks(rotation=45)
    
    # 子图2: 服务商效用对比
    plt.subplot(2, 3, 2)
    strategy_means = df.groupby('provider1_strategy')['total_provider_utility'].mean()
    strategy_means.plot(kind='bar', color='lightcoral')
    plt.title('不同策略下的平均服务商效用')
    plt.ylabel('服务商效用')
    plt.xticks(rotation=45)
    
    # 子图3: 总回报对比
    plt.subplot(2, 3, 3)
    strategy_means = df.groupby('provider1_strategy')['total_reward'].mean()
    strategy_means.plot(kind='bar', color='lightgreen')
    plt.title('不同策略下的平均总回报')
    plt.ylabel('总回报')
    plt.xticks(rotation=45)
    
    # 子图4: 热力图 - 策略组合效果
    plt.subplot(2, 3, 4)
    pivot_table = df.pivot_table(
        values='total_user_utility', 
        index='provider1_strategy', 
        columns='provider2_strategy', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('策略组合热力图 (用户效用)')
    
    # 子图5: 服务商效用分布
    plt.subplot(2, 3, 5)
    provider_utilities = []
    strategy_labels = []
    for strategy in ['honest', 'ours', 'worst', 'random']:
        for i in range(1, 4):
            col_name = f'provider_{i}_{strategy}'
            if col_name in df.columns:
                utilities = df[col_name].dropna()
                provider_utilities.extend(utilities)
                strategy_labels.extend([f'{strategy}'] * len(utilities))
    
    utility_df = pd.DataFrame({'strategy': strategy_labels, 'utility': provider_utilities})
    sns.boxplot(data=utility_df, x='strategy', y='utility')
    plt.title('各策略的服务商效用分布')
    plt.ylabel('服务商效用')
    plt.xticks(rotation=45)
    
    # 子图6: 用户vs服务商效用散点图
    plt.subplot(2, 3, 6)
    plt.scatter(df['total_provider_utility'], df['total_user_utility'], alpha=0.7)
    plt.xlabel('总服务商效用')
    plt.ylabel('总用户效用')
    plt.title('用户效用 vs 服务商效用')
    
    # 添加趋势线
    z = np.polyfit(df['total_provider_utility'], df['total_user_utility'], 1)
    p = np.poly1d(z)
    plt.plot(df['total_provider_utility'], p(df['total_provider_utility']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('simulation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pivot_table

def create_detailed_analysis(df):
    """创建详细分析图表"""
    
    # 重点分析我们的机制 vs 其他策略
    plt.figure(figsize=(20, 12))
    
    # 1. 我们的机制在不同对手下的表现
    plt.subplot(3, 4, 1)
    ours_scenarios = df[df['provider1_strategy'] == 'ours']
    if not ours_scenarios.empty:
        scenario_utilities = ours_scenarios.groupby(['provider2_strategy', 'provider3_strategy'])['total_user_utility'].mean()
        scenario_utilities.plot(kind='bar', color='gold')
        plt.title('我们的机制 vs 不同对手组合\n(用户效用)')
        plt.ylabel('用户效用')
        plt.xticks(rotation=45)
    
    # 2. 服务商效用对比
    plt.subplot(3, 4, 2)
    strategy_comparison = df.groupby('provider1_strategy')[['total_user_utility', 'total_provider_utility']].mean()
    strategy_comparison.plot(kind='bar', ax=plt.gca())
    plt.title('各策略平均效用对比')
    plt.ylabel('效用')
    plt.xticks(rotation=45)
    plt.legend(['用户效用', '服务商效用'])
    
    # 3. 效率分析 - 每单位委托的效用
    plt.subplot(3, 4, 3)
    df['user_utility_per_delegation'] = df['total_user_utility'] / df['total_delegations']
    df['provider_utility_per_delegation'] = df['total_provider_utility'] / df['total_delegations']
    
    efficiency_comparison = df.groupby('provider1_strategy')[['user_utility_per_delegation', 'provider_utility_per_delegation']].mean()
    efficiency_comparison.plot(kind='bar', ax=plt.gca())
    plt.title('每单位委托的效用效率')
    plt.ylabel('效用/委托次数')
    plt.xticks(rotation=45)
    plt.legend(['用户效率', '服务商效率'])
    
    # 4. 最佳服务商选择分析
    plt.subplot(3, 4, 4)
    best_provider_counts = df['best_provider'].value_counts()
    best_provider_counts.plot(kind='pie', autopct='%1.1f%%')
    plt.title('最佳服务商选择分布')
    plt.ylabel('')
    
    # 5-8. 各策略在不同位置的表现
    for i, position in enumerate(['provider1_strategy', 'provider2_strategy', 'provider3_strategy'], 1):
        plt.subplot(3, 4, 4 + i)
        position_utilities = df.groupby(position)['total_user_utility'].mean()
        position_utilities.plot(kind='bar', color=plt.cm.Set3(i))
        plt.title(f'位置{i}策略的用户效用')
        plt.ylabel('用户效用')
        plt.xticks(rotation=45)
    
    # 9. 策略稳定性分析
    plt.subplot(3, 4, 9)
    strategy_std = df.groupby('provider1_strategy')['total_user_utility'].std()
    strategy_std.plot(kind='bar', color='orange')
    plt.title('各策略的稳定性 (标准差)')
    plt.ylabel('用户效用标准差')
    plt.xticks(rotation=45)
    
    # 10. 委托次数分析
    plt.subplot(3, 4, 10)
    delegation_comparison = df.groupby('provider1_strategy')['total_delegations'].mean()
    delegation_comparison.plot(kind='bar', color='purple')
    plt.title('平均委托次数')
    plt.ylabel('委托次数')
    plt.xticks(rotation=45)
    
    # 11. 回报质量分析
    plt.subplot(3, 4, 11)
    df['avg_reward_per_delegation'] = df['total_reward'] / df['total_delegations']
    reward_quality = df.groupby('provider1_strategy')['avg_reward_per_delegation'].mean()
    reward_quality.plot(kind='bar', color='brown')
    plt.title('平均每委托的回报质量')
    plt.ylabel('回报/委托次数')
    plt.xticks(rotation=45)
    
    # 12. 综合评分
    plt.subplot(3, 4, 12)
    # 计算综合评分：用户效用 + 服务商效用 + 效率
    df['comprehensive_score'] = (
        df['total_user_utility'] / df['total_user_utility'].max() * 0.4 +
        df['total_provider_utility'] / df['total_provider_utility'].max() * 0.3 +
        df['user_utility_per_delegation'] / df['user_utility_per_delegation'].max() * 0.3
    )
    comprehensive_scores = df.groupby('provider1_strategy')['comprehensive_score'].mean()
    comprehensive_scores.plot(kind='bar', color='darkgreen')
    plt.title('综合评分')
    plt.ylabel('综合评分')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('detailed_simulation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df):
    """生成汇总报告"""
    print("=" * 60)
    print("模拟结果汇总报告")
    print("=" * 60)
    
    # 基本统计
    print(f"\n总模拟场景数: {len(df)}")
    print(f"策略类型: {df['provider1_strategy'].unique()}")
    
    # 各策略表现排名
    print("\n各策略平均用户效用排名:")
    user_utility_ranking = df.groupby('provider1_strategy')['total_user_utility'].mean().sort_values(ascending=False)
    for i, (strategy, utility) in enumerate(user_utility_ranking.items(), 1):
        print(f"{i}. {strategy}: {utility:.2f}")
    
    print("\n各策略平均服务商效用排名:")
    provider_utility_ranking = df.groupby('provider1_strategy')['total_provider_utility'].mean().sort_values(ascending=False)
    for i, (strategy, utility) in enumerate(provider_utility_ranking.items(), 1):
        print(f"{i}. {strategy}: {utility:.2f}")
    
    # 我们的机制表现
    if 'ours' in df['provider1_strategy'].values:
        ours_performance = df[df['provider1_strategy'] == 'ours']
        print(f"\n我们的机制表现:")
        print(f"  平均用户效用: {ours_performance['total_user_utility'].mean():.2f}")
        print(f"  平均服务商效用: {ours_performance['total_provider_utility'].mean():.2f}")
        print(f"  平均总回报: {ours_performance['total_reward'].mean():.2f}")
        print(f"  平均委托次数: {ours_performance['total_delegations'].mean():.1f}")
    
    # 最佳组合
    best_combination = df.loc[df['total_user_utility'].idxmax()]
    print(f"\n最佳用户效用组合: {best_combination['strategy_combination']}")
    print(f"  用户效用: {best_combination['total_user_utility']:.2f}")
    print(f"  服务商效用: {best_combination['total_provider_utility']:.2f}")
    
    return user_utility_ranking, provider_utility_ranking

def main():
    """主函数"""
    print("开始分析模拟结果...")
    
    # 加载所有结果
    all_results = load_all_results()
    print(f"成功加载 {len(all_results)} 个模拟结果")
    
    # 创建数据框
    df = create_summary_dataframe(all_results)
    print(f"数据框形状: {df.shape}")
    
    # 创建可视化
    print("创建可视化图表...")
    pivot_table = create_visualizations(df)
    
    # 创建详细分析
    print("创建详细分析图表...")
    create_detailed_analysis(df)
    
    # 生成报告
    print("生成汇总报告...")
    user_ranking, provider_ranking = generate_summary_report(df)
    
    # 保存数据
    df.to_csv('simulation_results_summary.csv', index=False)
    print("\n结果已保存到 simulation_results_summary.csv")
    print("图表已保存为 simulation_analysis.png 和 detailed_simulation_analysis.png")

if __name__ == "__main__":
    main()
