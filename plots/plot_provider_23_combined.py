import json
import os
import itertools
import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import MultipleLocator, FuncFormatter


CHOICES = [str(item) for item in list(range(6))]
LEGEND_LABELS = ["ours", "honest", "dishonest-model", "dishonest-length", "dishonest-all", "ours-honest-length"]

# Appearance controls
# Use a refined, publication-friendly palette (inspired by Nature/Science aesthetics)
# Provider 1, 2, 3 colors respectively (muted blue, green, red)
cmap = plt.get_cmap('Set2')
NATURE_PROVIDER_COLORS = {i: cmap(i % cmap.N) for i in range(1,4 )}


# Control the relative heights of upper (broken) vs lower axes per subplot.
# Increase this value to make the upper band taller relative to the lower band.
UPPER_TO_LOWER_HEIGHT_RATIO = 2.0


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
            all_scenarios = [f'0-{s1}-{s2}' for s1, s2 in others_sc]

        results = [json.load(open(RESULTS_PATH / item / 'result.json')) for item in all_scenarios]

        for result in results:
            provider_cost[strategy].append(result['providers'][i]['total_costs'])
            provider_price[strategy].append(result['providers'][i]['total_reported_price'])
            provider_reward[strategy].append(result['providers'][i]['total_rewards'])
            provider_utility[strategy].append(result['providers'][i]['total_provider_utility'])
            user_utility[strategy].append(result['providers'][i]['total_user_utility'])
            provider_delegations[strategy].append(result['providers'][i]['delegations'])
            provider_delta[strategy].append(result['providers'][i]['delta'])

        provider_results[strategy]['avg_cost'] = np.mean(provider_cost[strategy])
        provider_results[strategy]['avg_price'] = np.mean(provider_price[strategy])
        provider_results[strategy]['avg_reward'] = np.mean(provider_reward[strategy])
        provider_results[strategy]['provider_utility'] = np.mean(provider_utility[strategy])
        provider_results[strategy]['user_utility'] = np.mean(user_utility[strategy])
        provider_results[strategy]['delegations'] = np.mean(provider_delegations[strategy])
        provider_results[strategy]['delta'] = np.mean(provider_delta[strategy])

    return provider_results


def plot_providers_combined(provider2_results, provider3_results, provider1_results, save_path, choices=CHOICES):
    os.makedirs(save_path, exist_ok=True)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')

    # Fine-tune figure aesthetics for a polished, publication-ready look
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.edgecolor": "#2F3E46",
        "axes.linewidth": 1.0,
        "grid.alpha": 0.85,
        "grid.color": "#D0D0D0",
        "grid.linewidth": 0.6,
    })

    # Provider-level colors (consistent across strategies)
    provider_colors = NATURE_PROVIDER_COLORS

    metrics = [
        ('provider_utility', 'Average Provider Utility'),
        ('user_utility', 'Average User Utility'),
        ('delegations', 'Average Delegations'),
    ]

    # Convert results to arrays for both providers
    data_p2 = {key: [float(provider2_results[strategy][key]) for strategy in choices] for key, _ in metrics}
    data_p3 = {key: [float(provider3_results[strategy][key]) for strategy in choices] for key, _ in metrics}

    fig = plt.figure(figsize=(18, 5.6))
    gs = fig.add_gridspec(1, 3)

    x = np.arange(len(choices))
    bar_width = 0.36
    x2 = x - bar_width / 2
    x3 = x + bar_width / 2

    # Manual broken-axis bands per metric
    upper_low = [
        {"low": (0, 2000, 1000), "up": (163000, 170000, 2500)},
        {"low": (0, 14000, 5000), "up": (650000, 658000, 3000)},
        {"low": (0, 24000, 8000), "up": (914000, 924000, 4000)},
    ]

    for idx, (metric_key, metric_label) in enumerate(metrics):
        low0, low1, low_step = upper_low[idx]['low']
        up0, up1, up_step = upper_low[idx]['up']

        # Upper band is taller than lower band for clearer emphasis on large values
        sub_gs = gs[0, idx].subgridspec(2, 1, height_ratios=[UPPER_TO_LOWER_HEIGHT_RATIO, 1.0], hspace=0.02)
        ax_up = fig.add_subplot(sub_gs[0])
        ax_low = fig.add_subplot(sub_gs[1], sharex=ax_up)

        # Plot both providers on both axes
        ax_low.bar(x2, data_p2[metric_key], width=bar_width, color=provider_colors[2], edgecolor='black', linewidth=0.6)
        ax_low.bar(x3, data_p3[metric_key], width=bar_width, color=provider_colors[3], edgecolor='black', linewidth=0.6)
        ax_up.bar(x2, data_p2[metric_key], width=bar_width, color=provider_colors[2], edgecolor='black', linewidth=0.6)
        ax_up.bar(x3, data_p3[metric_key], width=bar_width, color=provider_colors[3], edgecolor='black', linewidth=0.6)

        # Y limits and ticks
        ax_low.set_ylim(low0, low1)
        ax_up.set_ylim(up0, up1)
        ax_low.yaxis.set_major_locator(MultipleLocator(low_step))
        ax_up.yaxis.set_major_locator(MultipleLocator(up_step))
        # Middle subplot in millions on both bands for consistent appearance
        if idx == 1:
            fmt_million = FuncFormatter(lambda y, pos: f"{y/1e6:.3f}")
            ax_low.yaxis.set_major_formatter(fmt_million)
            ax_up.yaxis.set_major_formatter(fmt_million)

        # Hide the first (lowest) horizontal tick on the upper broken axis
        try:
            yticks_up = list(ax_up.get_yticks())
            filtered_up = [t for t in yticks_up if t > up0 + 1e-9]
            if len(filtered_up) >= 1:
                ax_up.set_yticks(filtered_up)
        except Exception:
            pass

        # Tighten vertical margins to reduce whitespace
        ax_low.margins(y=0.02)
        ax_up.margins(y=0.02)

        # Titles and x ticks
        ax_up.set_title(metric_label, fontsize=14, pad=8)
        ticklabels = [f"{LEGEND_LABELS[i]}" for i in range(len(choices))]
        ax_low.set_xticks(x)
        ax_low.set_xticklabels(ticklabels, rotation=30, ha='right', fontsize=12)
        ax_up.tick_params(labelbottom=False)
        ax_low.tick_params(axis='y', labelsize=11)
        ax_up.tick_params(axis='y', labelsize=11)

        # For the second subplot, add a small '1e6' label at the very top of the upper y-axis
        if idx == 1:
            try:
                ax_up.text(-0.06, 1.02, '1e6', transform=ax_up.transAxes, ha='left', va='bottom', fontsize=10)
            except Exception:
                pass

        # Broken markers near y-axis
        ax_low.spines['top'].set_visible(False)
        ax_up.spines['bottom'].set_visible(False)

        # Draw Provider 1 bar before annotations to avoid covering labels
        try:
            ours_idx = choices.index('0')
            p1_val = float(provider1_results['0'][metric_key])
            # Place Provider 1 as the leftmost bar in the 'ours' group with equal spacing
            x_p1 = x[ours_idx] - 1.5 * bar_width
            p1_width = bar_width * 0.7
            # Draw on both bands if needed to avoid hollow base across the break
            ann_ax_p1 = None
            if p1_val >= up0:
                ax_low.bar(x_p1, low1, width=p1_width, color=provider_colors[1], edgecolor='black', linewidth=0.6, zorder=3)
                ax_up.bar(x_p1, p1_val - up0, bottom=up0, width=p1_width, color=provider_colors[1], edgecolor='black', linewidth=0.6, zorder=3)
                ann_ax_p1 = ax_up
            elif p1_val <= low1:
                ax_low.bar(x_p1, p1_val, width=p1_width, color=provider_colors[1], edgecolor='black', linewidth=0.6, zorder=3)
                ann_ax_p1 = ax_low
            else:
                ax_low.bar(x_p1, p1_val, width=p1_width, color=provider_colors[1], edgecolor='black', linewidth=0.6, zorder=3)
                ann_ax_p1 = ax_low
            # if ann_ax_p1 is not None:
            #     # Customize Provider 1 annotation per subplot
            #     if idx == 1:
            #         ann_ax_p1.annotate("5.27×1e6", xy=(x_p1, p1_val), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, zorder=10)
            #     elif idx == 2:
            #         ann_ax_p1.annotate(f"{p1_val:.2f}", xy=(x_p1, p1_val), xytext=(6, 3), textcoords="offset points", ha='left', va='bottom', fontsize=9, zorder=10)
            #     else:
            #         ann_ax_p1.annotate(f"{p1_val:.2f}", xy=(x_p1, p1_val), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, zorder=10)
        except Exception:
            pass

        # Annotate only the global maximum bar among both providers
        all_vals = np.array(data_p2[metric_key] + data_p3[metric_key])
        max_val = float(np.max(all_vals)) if all_vals.size else None
        if max_val is not None:
            # Decide provider and index for annotation
            if max_val in data_p2[metric_key]:
                ann_idx = data_p2[metric_key].index(max_val)
                ann_x = x2[ann_idx]
            else:
                ann_idx = data_p3[metric_key].index(max_val)
                ann_x = x3[ann_idx]
            ann_ax = ax_low if max_val <= low1 else ax_up if max_val >= up0 else (ax_low if abs(max_val - low1) < abs(max_val - up0) else ax_up)
            # If the max is Provider 2 at 'ours', nudge right to avoid Provider 1 bar; ensure on top via zorder
            ours_idx_val = choices.index('0')
            # is_p2_ours = (max_val in data_p2[metric_key] and ann_idx == ours_idx_val)
            # if is_p2_ours:
            #     ann_ax.annotate(f"{max_val:.2f}", xy=(ann_x, max_val), xytext=(8, 4), textcoords="offset points", ha='left', va='bottom', fontsize=9, zorder=10)
            # else:
            #     ann_ax.annotate(f"{max_val:.2f}", xy=(ann_x, max_val), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, zorder=10)


    fig.suptitle("Providers 2 and 3 Performance by Strategy", fontsize=16, y=0.98)

    # Legend: Provider colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=provider_colors[1], edgecolor='black', label='Provider 1 (Ours)'),
        Patch(facecolor=provider_colors[2], edgecolor='black', label='Provider 2'),
        Patch(facecolor=provider_colors[3], edgecolor='black', label='Provider 3')
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=13)

    # Layout and save
    plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.24)
    outfile = save_path / 'providers_2_3_combined.pdf'
    fig.canvas.draw()
    fig.savefig(outfile, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/toy_game/default.yaml')
    parser.add_argument('--output-dir', type=str, default='outputs/toy_game/v1')
    args = parser.parse_args()

    CONFIG_PATH = Path(args.config)
    global RESULTS_PATH
    RESULTS_PATH = Path(args.output_dir)

    # Load config (not directly used, but kept for parity)
    _ = yaml.safe_load(open(CONFIG_PATH))

    provider2_results = get_result(1)
    provider3_results = get_result(2)
    provider1_results = get_result(0)
    print(provider1_results['0'])

    plot_providers_combined(provider2_results, provider3_results, provider1_results, save_path=RESULTS_PATH / 'figs')


