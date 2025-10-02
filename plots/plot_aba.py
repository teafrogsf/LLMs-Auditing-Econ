import json
import os
import itertools
from random import choice
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

CHOICES = [str(item) for item in list(range(6))]
LEGEND_LABELS = ["ours", "honest", "dishonest-model", "dishonest-token", "dishonest-all", "ours-honest-token"]


def plot_histogram(num_provider, provider_results, save_path, choices=CHOICES):
    os.makedirs(save_path, exist_ok=True)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')

    cmap = plt.get_cmap('tab10')
    strategy_colors = [cmap(i % 10) for i in range(len(choices))]

    metrics = [
        ('provider_utility', 'Average Provider Utility'),
        ('user_utility', 'Average User Utility'),
        ('delegations', 'Average Delegations'),
        ('delta', 'Average Delta'),
    ]

    data_by_metric = {
        key: [provider_results[strategy][key] for strategy in choices]
        for key, _ in metrics
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    x = np.arange(len(choices))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        y_vals = data_by_metric[metric_key]
        bars = ax.bar(x, y_vals, color=strategy_colors, edgecolor='black', linewidth=0.6)

        ax.set_title(metric_label, fontsize=16, pad=10)
        ax.set_ylabel(metric_label, fontsize=14)
        ax.set_xticks(x)
        # Strategies are 0-based in folders; display 1-based labels in plots
        ax.set_xticklabels([f"{LEGEND_LABELS[i]}" for i in range(len(choices))], rotation=45, ha='right', fontsize=13)
        ax.tick_params(axis='y', labelsize=13)
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

    fig.suptitle(f"Provider {num_provider + 1} Performance by Strategy", fontsize=18, y=0.98)

    # Shared legend for strategies
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=strategy_colors[i], edgecolor='black', label=LEGEND_LABELS[i])
                      for i in range(len(choices))]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=12)

    fig.tight_layout(rect=[0.02, 0.08, 1, 0.95])

    outfile = save_path / f"provider_{num_provider + 1}.pdf"
    fig.savefig(outfile, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--config', type=str, default='config/toy_game/default.yaml')
    paser.add_argument('--output-dir', type=str, default='outputs/toy_game/v1')
    args = paser.parse_args()
    CONFIG_PATH = Path(args.config)
    RESULTS_PATH = Path(args.output_dir)
    scenarios = [f'{item}-1-1' for item in CHOICES]
    config = yaml.safe_load(open(CONFIG_PATH))



    num_providers = len(config['providers'])
    num_others = num_providers - 1


    for i in range(1):

        provider_cost = {k: [] for k in CHOICES}
        provider_reward = {k: [] for k in CHOICES}
        provider_price = {k: [] for k in CHOICES}
        provider_utility = {k: [] for k in CHOICES} 
        user_utility = {k: [] for k in CHOICES}
        provider_results = {k: {} for k in CHOICES}
        provider_delegations = {k: [] for k in CHOICES}
        provider_delta = {k: [] for k in CHOICES}
        for strategy in CHOICES:
            
            others_sc = itertools.product(CHOICES, repeat=2)
            all_scenarios = [f'{strategy}-1-1']


            print(all_scenarios)
            results = [json.load(open(RESULTS_PATH / item / 'result.json')) for item in all_scenarios]

            
            for result in results:
                provider_cost[strategy].append(result['providers'][i]['total_costs']) 
                provider_price[strategy].append(result['providers'][i]['total_reported_price']) 
                provider_reward[strategy].append(result['providers'][i]['total_rewards']) 
                provider_utility[strategy].append(result['providers'][i]['total_provider_utility']) 
                user_utility[strategy].append(result['providers'][i]['total_user_utility']) 
                provider_delegations[strategy].append(result['providers'][i]['delegations'])
                provider_delta[strategy].append(result['providers'][i]['delta'])
            provider_results[strategy]['avg_cost'] =  np.mean(provider_cost[strategy])
            provider_results[strategy]['avg_price'] =  np.mean(provider_price[strategy])
            provider_results[strategy]['avg_reward'] =  np.mean(provider_reward[strategy])
            provider_results[strategy]['provider_utility'] =  np.mean(provider_utility[strategy])
            provider_results[strategy]['user_utility'] =  np.mean(user_utility[strategy])
            provider_results[strategy]['delegations'] =  np.mean(provider_delegations[strategy])
            provider_results[strategy]['delta'] =  np.mean(provider_delta[strategy])
        plot_histogram(i, provider_results, save_path=RESULTS_PATH / 'figs')


            