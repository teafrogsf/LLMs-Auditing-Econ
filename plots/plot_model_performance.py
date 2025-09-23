import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path




model_info = yaml.safe_load(open('config/nl_graph/model_config.yaml'))
for model in model_info:
    
    model_info[model]['cost_mu'] = model_info[model]['output_tokens_mu'] * model_info[model]['output_token_price']
    model_info[model]['utility_mu'] = model_info[model]['score_mu'] * 5 - model_info[model]['cost_mu']

def plot_utility_vs_cost(model_info_dict):
    names = list(model_info_dict.keys())
    utilities = [model_info_dict[name]['utility_mu'] for name in names]
    costs = [model_info_dict[name]['cost_mu'] for name in names]

    # Scientific aesthetic
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.2)
    palette = sns.color_palette('icefire', n_colors=len(names))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {name: palette[i] for i, name in enumerate(names)}
    for name, u, c in zip(names, utilities, costs):
        ax.scatter(c, u, s=70, c=[colors[name]], edgecolors='black', linewidths=0.6, alpha=0.9)
        ax.annotate(name, (c, u), xytext=(5, 5), textcoords='offset points', fontsize=9, color=colors[name])

    ax.set_xlabel('cost_mu')
    ax.set_ylabel('utility_mu')
    ax.set_title('Model Utility vs Cost')
    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

    out_dir = Path('outputs/nl_graph')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'utility_vs_cost.pdf'
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')


if __name__ == '__main__':
    plot_utility_vs_cost(model_info)
