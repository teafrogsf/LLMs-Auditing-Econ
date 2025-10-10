import json
import os
import itertools
from random import choice
from webbrowser import get
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import math
from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter


CHOICES = [str(item) for item in list(range(6))]
LEGEND_LABELS = ["ours", "honest", "dishonest-model", "dishonest-token", "dishonest-all", "ours-honest-token"]


def plot_histogram(num_provider, provider_results, save_path, choices=CHOICES, provider1_result=None):
    os.makedirs(save_path, exist_ok=True)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')

    cmap = plt.get_cmap('Set2')
    strategy_colors = [cmap(i % cmap.N) for i in range(len(choices))]

    metrics = [
        ('provider_utility', 'Average Provider Utility'),
        ('user_utility', 'Average User Utility'),
        ('delegations', 'Average Delegations'),
    ]

    data_by_metric = {
        key: [float(provider_results[strategy][key]) for strategy in choices]
        for key, _ in metrics
    }
    print(data_by_metric)
    # print(provider1_result)
    # exit()

    # Try import brokenaxes once
    try:
        from brokenaxes import brokenaxes as BrokenAxes
        _brokenaxes_available = True
    except Exception:
        BrokenAxes = None
        _brokenaxes_available = False

    fig = plt.figure(figsize=(18, 5.0))
    gs = fig.add_gridspec(1, 3)

    x = np.arange(len(choices))
    upper_low = [
        {
            "low": (0, 5000, 2000), 
            "up": (43000, 46000, 1000)
        },
        {
            "low": (0, 100000, 30000), 
            "up": (5270000, 5276000, 2000)
        },
        {
            "low": (0, 17000, 5000), 
            "up": (920000, 925000, 2500)
        },
        ]

    for idx, (metric_key, metric_label) in enumerate(metrics):
        y_vals = data_by_metric[metric_key]

        # # Determine baseline if provided
        baseline_val = float(provider1_result['0'][metric_key])
    
        print(y_vals)
        print(baseline_val)

        # max_bar = float(np.nanmax(y_vals)) if len(y_vals) > 0 else 0.0
        # lower_upper = max_bar * 1.10
        # # Ensure minimal sensible lower band upper bound
        # if lower_upper <= 0:
        #     lower_upper = 1.0

        use_broken = True

        if use_broken:
            # Manual two-axes broken view to allow precise control of the gap and markers
            low0, low1, low_step = upper_low[idx]['low']
            up0, up1, up_step = upper_low[idx]['up']

            # Make the upper band shorter and also compress the lower band
            # Tip: increase the second value to make lower taller; decrease it to compress
            sub_gs = gs[0, idx].subgridspec(2, 1, height_ratios=[0.8, 1.2], hspace=0.05)
            # Place the UPPER band on the TOP axes, LOWER band on the BOTTOM axes
            ax_up = fig.add_subplot(sub_gs[0])
            ax_low = fig.add_subplot(sub_gs[1], sharex=ax_up)

            # Plot bars on both axes
            ax_low.bar(x, y_vals, color=strategy_colors, edgecolor='black', linewidth=0.6)
            ax_up.bar(x, y_vals, color=strategy_colors, edgecolor='black', linewidth=0.6)

            # Y limits and ticks
            ax_low.set_ylim(low0, low1)
            ax_up.set_ylim(up0, up1)
            ax_low.yaxis.set_major_locator(MultipleLocator(low_step))
            ax_up.yaxis.set_major_locator(MultipleLocator(up_step))
            # For the middle subplot (user_utility), make lower axis use the same million-scale format as upper
            if idx == 1:
                fmt_million = FuncFormatter(lambda y, pos: f"{y/1e6:.3f}")
                ax_low.yaxis.set_major_formatter(fmt_million)
            # Tighten vertical margins to reduce extra whitespace
            ax_low.margins(y=0.03)
            ax_up.margins(y=0.02)

            # Title on upper axis; x ticks on lower axis only
            ax_up.set_title(metric_label, fontsize=14, pad=10)
            ticklabels = [f"{LEGEND_LABELS[i]}" for i in range(len(choices))]
            ax_low.set_xticks(x)
            ax_low.set_xticklabels(ticklabels, rotation=30, ha='right', fontsize=12)
            ax_up.tick_params(labelbottom=False)
            # Y tick label sizes
            ax_low.tick_params(axis='y', labelsize=11)
            ax_up.tick_params(axis='y', labelsize=11)

            # Clean spines and draw break marks near y-axis
            ax_low.spines['top'].set_visible(False)
            ax_up.spines['bottom'].set_visible(False)
            # Slashes on low axis top
            kwargs_low = dict(transform=ax_low.transAxes, color='k', clip_on=False, linewidth=1.0)
            ax_low.plot((-0.012, +0.012), (1.00, 1.02), **kwargs_low)
            ax_low.plot((+0.012, -0.012), (1.00, 1.02), **kwargs_low)
            # Slashes on upper axis bottom
            kwargs_up = dict(transform=ax_up.transAxes, color='k', clip_on=False, linewidth=1.0)
            ax_up.plot((-0.012, +0.012), (0.00, -0.02), **kwargs_up)
            ax_up.plot((+0.012, -0.012), (0.00, -0.02), **kwargs_up)

            # Annotate only the maximum bar value on the appropriate axis
            max_idx = int(np.argmax(y_vals)) if len(y_vals) else None
            if max_idx is not None:
                max_val = y_vals[max_idx]
                if max_val <= low1:
                    ann_ax = ax_low
                elif max_val >= up0:
                    ann_ax = ax_up
                else:
                    ann_ax = ax_low if abs(max_val - low1) < abs(max_val - up0) else ax_up
                ann_ax.annotate(f"{max_val:.2f}",
                                xy=(max_idx, max_val),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)

            # Baseline line and label on upper axis if within upper band
            if baseline_val is not None and up0 <= baseline_val <= up1:
                ax_up.hlines(y=baseline_val, xmin=-0.5, xmax=len(choices) - 0.5,
                             colors='red', linestyles='--', linewidth=1.2)
                right_x = len(choices) - 0.52
                ax_up.annotate(f"{baseline_val:.2f}",
                               xy=(right_x, baseline_val),
                               xytext=(0, -6),
                               textcoords="offset points",
                               color='red', fontsize=10,
                               ha='right', va='top',
                               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='red', alpha=0.7),
                               clip_on=True)

    # No extra subplot to hide when using gridspec

    fig.suptitle(f"Provider {num_provider + 1} Performance by Strategy", fontsize=16, y=0.98)

    # Legend: only keep Provider 1 baseline
    from matplotlib.lines import Line2D
    legend_handles = []
    if provider1_result is not None:
        legend_handles.append(Line2D([0], [0], color='red', linestyle='--', label='Provider 1 (Ours)'))
    if legend_handles:
        fig.legend(handles=legend_handles, loc='lower center', ncol=1, frameon=False,
                   bbox_to_anchor=(0.5, -0.06), fontsize=11)

    # Adjust margins to avoid overlaps (axes lowered to clear suptitle; extra space for legend below)
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.24)

    outfile = save_path / f"provider_{num_provider + 1}.pdf"
    fig.canvas.draw()

    fig.savefig(outfile, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

def get_result(i):
    provider_cost = {k: [] for k in CHOICES}
    provider_reward = {k: [] for k in CHOICES}
    provider_price = {k: [] for k in CHOICES}
    provider_utility = {k: [] for k in CHOICES} 
    user_utility = {k: [] for k in CHOICES}
    provider_results = {k: {} for k in CHOICES}
    provider_delegations = {k: [] for k in CHOICES}
    provider_delta = {k: [] for k in CHOICES}
    for strategy in CHOICES:
        
        others_sc = CHOICES
        if i == 1:
            all_scenarios = [f'0-{strategy}-{s}' for s in others_sc]
        elif i == 2:
            all_scenarios = [f'0-{s}-{strategy}' for s in others_sc]
        elif i == 0:
            others_sc = itertools.product(CHOICES, repeat=2)
            all_scenarios = [f'0-{s1}-{s2}' for s1,s2 in others_sc]
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

    provider_results['provider_utility'] = [provider_results[strategy]['provider_utility'] for strategy in CHOICES]
    provider_results['user_utility'] = [provider_results[strategy]['user_utility'] for strategy in CHOICES]
    provider_results['delegations'] = [provider_results[strategy]['delegations'] for strategy in CHOICES]
        # provider_results['provider_utility'] = [provider_results[strategy]['provider_utility'] for strategy in CHOICES]
        

        

      

    return provider_results

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--config', type=str, default='config/toy_game/default.yaml')
    paser.add_argument('--output-dir', type=str, default='outputs/toy_game/v1')
    args = paser.parse_args()
    CONFIG_PATH = Path(args.config)
    RESULTS_PATH = Path(args.output_dir)

    config = yaml.safe_load(open(CONFIG_PATH))
    num_providers = len(config['providers'])
    # num_others = num_providers - 1
    provider_results = get_result(1)
    json.dump(provider_results, open(RESULTS_PATH / 'figs' / 'provider2.json', 'w'), indent=2)
    provider1_results = get_result(0)

    
    plot_histogram(1, provider_results, save_path=RESULTS_PATH / 'figs', provider1_result=provider1_results)


    provider_idx = 2
    provider_results = get_result(provider_idx)
  
    plot_histogram(provider_idx, provider_results, save_path=RESULTS_PATH / 'figs', provider1_result=provider1_results)
    json.dump(provider1_results, open(RESULTS_PATH / 'figs' / 'provider1.json', 'w'), indent=2)
   
    json.dump(provider_results, open(RESULTS_PATH / 'figs' / 'provider3.json', 'w'), indent=2)