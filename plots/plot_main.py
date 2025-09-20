import json
import os
import itertools
from random import choice
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

CHOICES = ['honest', 'ours', 'worst', 'random', 'h1w2', 'w1h2']
RESULTS_PATH = Path('outputs/default')
CONFIG_PATH = Path('config/default.yaml')

def plot_histogram(num_provider, provider_results, choices=CHOICES, save_path=RESULTS_PATH / 'figs'):
    os.makedirs(save_path, exist_ok=True)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')

    cmap = plt.get_cmap('tab10')
    strategy_colors = [cmap(i % 10) for i in range(len(choices))]

    metrics = [
        ('avg_cost', 'Average Cost'),
        ('avg_price', 'Average Price'),
        ('avg_reward', 'Average Reward'),
        ('provider_utility', 'Average Provider Utility'),
        ('user_utility', 'Average User Utility'),
    ]

    data_by_metric = {
        key: [provider_results[strategy][key] for strategy in choices]
        for key, _ in metrics
    }

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    x = np.arange(len(choices))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        y_vals = data_by_metric[metric_key]
        bars = ax.bar(x, y_vals, color=strategy_colors, edgecolor='black', linewidth=0.6)

        ax.set_title(metric_label, fontsize=14, pad=10)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([c for c in choices], rotation=25, ha='right', fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.margins(y=0.2)

        # Annotate bar values
        for rect, val in zip(bars, y_vals):
            height = rect.get_height()
            ax.annotate(f"{val:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    # Hide any extra subplot if grid is larger than metrics
    if len(axes) > len(metrics):
        axes[-1].set_visible(False)

    fig.suptitle(f"Provider {num_provider + 1} Performance by Strategy", fontsize=16, y=0.98)

    # Shared legend for strategies
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=strategy_colors[i], edgecolor='black', label=choices[i])
                      for i in range(len(choices))]
    fig.legend(handles=legend_handles, loc='lower center', ncol=min(6, len(choices)), frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=11)

    fig.tight_layout(rect=[0.02, 0.06, 1, 0.95])

    outfile = save_path / f"provider_{num_provider + 1}.pdf"
    fig.savefig(outfile, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    scenarios = list(itertools.product(CHOICES, repeat=3))
    config = yaml.safe_load(open(CONFIG_PATH))

    print(config)

    num_providers = len(config['providers'])
    num_others = num_providers - 1

    for i in range(num_providers):
        provider_cost = {k: [] for k in CHOICES}
        provider_reward = {k: [] for k in CHOICES}
        provider_price = {k: [] for k in CHOICES}
        provider_utility = {k: [] for k in CHOICES} 
        user_utility = {k: [] for k in CHOICES}
        provider_results = {k: {} for k in CHOICES}
        for strategy in CHOICES:
            all_scenarios = list(itertools.product(CHOICES, repeat=num_others))
            for idx, item in enumerate(all_scenarios):
                item = list(item)
                item.insert(i, strategy)
                all_scenarios[idx] = "-".join(item)


            results = [json.load(open(RESULTS_PATH / item / 'result.json')) for item in all_scenarios]

            
            for result in results:
                provider_cost[strategy].append(result['providers'][i]['total_cost']) 
                provider_price[strategy].append(result['providers'][i]['total_price']) 
                provider_reward[strategy].append(result['providers'][i]['total_reward']) 
                provider_utility[strategy].append(result['providers'][i]['provider_utility']) 
                user_utility[strategy].append(result['providers'][i]['user_utility']) 

            provider_results[strategy]['avg_cost'] =  np.mean(provider_cost[strategy])
            provider_results[strategy]['avg_price'] =  np.mean(provider_price[strategy])
            provider_results[strategy]['avg_reward'] =  np.mean(provider_reward[strategy])
            provider_results[strategy]['provider_utility'] =  np.mean(provider_utility[strategy])
            provider_results[strategy]['user_utility'] =  np.mean(user_utility[strategy])

        plot_histogram(i, provider_results)


            