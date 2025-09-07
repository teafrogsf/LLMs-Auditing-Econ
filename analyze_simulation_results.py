import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paper-style plotting defaults (English only)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STIXGeneral']
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', palette='colorblind')

CHOICES = ['honest', 'ours', 'worst', 'random']


def parse_log_file(log_file_path):
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        filename = os.path.basename(log_file_path)
        strategy_match = re.search(r'simulator_(.+)\.log', filename)
        if not strategy_match:
            return None
        strategies = strategy_match.group(1).split('-')
        results = {}
        total_time_match = re.search(r'总时间步数：(\d+)', content)
        total_delegations_match = re.search(r'实际委托次数：(\d+)', content)
        best_provider_match = re.search(r'最佳服务商：(\d+)', content)
        if total_time_match:
            results['total_time'] = int(total_time_match.group(1))
        if total_delegations_match:
            results['total_delegations'] = int(total_delegations_match.group(1))
        if best_provider_match:
            results['best_provider'] = int(best_provider_match.group(1))
        provider_stats = {}
        provider_pattern = (
            r'服务商(\d+):\s*\n.*?委托次数：(\d+)\s*\n.*?总价格：([\d.-]+)\s*\n.*?总成本：([\d.-]+)\s*\n'
            r'.*?服务商效用：([\d.-]+)\s*\n.*?总回报：([\d.-]+)\s*\n.*?平均回报：([\d.-]+)\s*\n.*?用户效用：([\d.-]+)'
        )
        for match in re.finditer(provider_pattern, content, re.DOTALL):
            pid = int(match.group(1))
            provider_stats[pid] = {
                'delegations': int(match.group(2)),
                'total_price': float(match.group(3)),
                'total_cost': float(match.group(4)),
                'provider_utility': float(match.group(5)),
                'total_reward': float(match.group(6)),
                'avg_reward': float(match.group(7)),
                'user_utility': float(match.group(8)),
            }
        results['provider_stats'] = provider_stats
        results['strategies'] = strategies
        return results
    except Exception as e:
        print(f"Error parsing {log_file_path}: {e}")
        return None


def load_all_results():
    logs_dir = Path('logs')
    all_results = []
    for log_file in logs_dir.glob('simulator_*.log'):
        result = parse_log_file(log_file)
        if result:
            all_results.append(result)
    return all_results


def create_summary_dataframe(all_results):
    rows = []
    for result in all_results:
        strategies = result['strategies']
        provider_stats = result['provider_stats']
        total_user_utility = sum(s['user_utility'] for s in provider_stats.values())
        total_provider_utility = sum(s['provider_utility'] for s in provider_stats.values())
        total_reward = sum(s['total_reward'] for s in provider_stats.values())
        row = {
            'strategy_combination': '-'.join(strategies),
            'provider1_strategy': strategies[0],
            'provider2_strategy': strategies[1],
            'provider3_strategy': strategies[2],
            'total_user_utility': total_user_utility,
            'total_provider_utility': total_provider_utility,
            'total_reward': total_reward,
            'total_delegations': result.get('total_delegations', np.nan),
            'best_provider': result.get('best_provider', np.nan),
        }
        # columns for provider i utility under strategy s
        for i, s in enumerate(strategies, 1):
            if i in provider_stats:
                row[f'provider_{i}_{s}'] = provider_stats[i]['provider_utility']
        # best provider strategy
        if 'best_provider' in result and not pd.isna(result['best_provider']):
            bp = int(result['best_provider'])
            if 1 <= bp <= 3:
                row['best_provider_strategy'] = strategies[bp - 1]
        rows.append(row)
    return pd.DataFrame(rows)


def save_fig(fig, basename):
    png = f"{basename}.png"
    pdf = f"{basename}.pdf"
    eps = f"{basename}.eps"
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(eps, bbox_inches='tight')


# Figure 1: Average utility per provider (three subplots)

def plot_figure1_provider_utilities(df):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharey=True)
    for i in range(3):
        ax = axes[i]
        means = []
        labels = []
        for s in CHOICES:
            col = f'provider_{i+1}_{s}'
            if col in df.columns:
                means.append(df[col].mean())
                labels.append(s)
        bars = sns.barplot(x=labels, y=means, ax=ax)
        # 在柱状图上添加数值标签
        for j, (bar, mean_val) in enumerate(zip(bars.patches, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{mean_val:.2f}', ha='center', va='bottom', fontsize=8)
        ax.set_title(f'Provider {i+1} Average Utility')
        ax.set_ylabel('Average Utility' if i == 0 else '')
        ax.set_xlabel('Strategy')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('Average Utility per Provider (by strategy)', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig1_provider_utilities')
    plt.close(fig)


# Figure 2: User average utility and total reward heatmaps across all combinations

def plot_figure2_user_heatmaps(df):
    fig, axes = plt.subplots(2, len(CHOICES), figsize=(4*len(CHOICES), 7.2))
    for j, s1 in enumerate(CHOICES):
        sub = df[df['provider1_strategy'] == s1]
        if sub.empty:
            continue
        pivot_u = sub.pivot_table(values='total_user_utility', index='provider2_strategy', columns='provider3_strategy', aggfunc='mean')
        pivot_r = sub.pivot_table(values='total_reward', index='provider2_strategy', columns='provider3_strategy', aggfunc='mean')
        pivot_u = pivot_u.reindex(index=CHOICES, columns=CHOICES)
        pivot_r = pivot_r.reindex(index=CHOICES, columns=CHOICES)
        sns.heatmap(pivot_u, ax=axes[0, j], annot=True, fmt='.0f', cmap='YlGnBu', cbar=j==len(CHOICES)-1)
        axes[0, j].set_title(f'User Avg. Utility | Provider1={s1}')
        axes[0, j].set_xlabel('Provider3 Strategy')
        axes[0, j].set_ylabel('Provider2 Strategy')
        sns.heatmap(pivot_r, ax=axes[1, j], annot=True, fmt='.0f', cmap='YlOrRd', cbar=j==len(CHOICES)-1)
        axes[1, j].set_title(f'User Total Reward | Provider1={s1}')
        axes[1, j].set_xlabel('Provider3 Strategy')
        axes[1, j].set_ylabel('Provider2 Strategy')
    plt.tight_layout()
    save_fig(fig, 'fig2_user_heatmaps')
    plt.close(fig)


# Figure 3: Additional results (best provider strategy distribution; user vs provider utility)

def plot_figure3_extra(df):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
    # Left: distribution of best provider strategies
    if 'best_provider_strategy' in df.columns:
        counts = df['best_provider_strategy'].value_counts().reindex(CHOICES)
        counts = counts.fillna(0)
        sns.barplot(x=counts.index, y=counts.values, ax=axes[0])
        axes[0].set_title('Distribution of Best-Provider Strategies')
        axes[0].set_ylabel('Count')
        axes[0].set_xlabel('Strategy')
        axes[0].tick_params(axis='x', rotation=30)
    # Right: scatter (color = number of "ours" providers)
    num_ours = df[['provider1_strategy', 'provider2_strategy', 'provider3_strategy']].apply(lambda r: sum(1 for x in r if x == 'ours'), axis=1)
    sc = axes[1].scatter(df['total_provider_utility'], df['total_user_utility'], c=num_ours, cmap='viridis', alpha=0.85, edgecolor='none')
    axes[1].set_xlabel('Total Provider Utility')
    axes[1].set_ylabel('Total User Utility')
    axes[1].set_title('User vs Provider Utility (color = # of "ours")')
    cbar = plt.colorbar(sc, ax=axes[1])
    cbar.set_label('Number of "ours" providers')
    fig.tight_layout()
    save_fig(fig, 'fig3_extra')
    plt.close(fig)


def main():
    print('Loading logs...')
    results = load_all_results()
    print(f'Loaded {len(results)} logs')
    if not results:
        print('No logs found. Exit.')
        return
    df = create_summary_dataframe(results)
    df.to_csv('simulation_results_summary.csv', index=False)
    print('Generating Figure 1...')
    plot_figure1_provider_utilities(df)
    print('Generating Figure 2...')
    plot_figure2_user_heatmaps(df)
    print('Generating Figure 3...')
    plot_figure3_extra(df)
    print('Done. Saved: fig1_provider_utilities.(png/pdf/eps), fig2_user_heatmaps.(png/pdf/eps), fig3_extra.(png/pdf/eps)')


if __name__ == '__main__':
    main()
