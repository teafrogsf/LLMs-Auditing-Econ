import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_scatter():
    sns.set_theme(context="talk", style="whitegrid")

    items = list(plot_data.items())
    if len(items) == 0:
        return

    colors = sns.color_palette("mako", n_colors=len(items))

    x_values = [vals["miu_l"] for _, vals in items]
    y_values = [vals["miu_r"] for _, vals in items]

    fig, ax = plt.subplots(figsize=(9, 7))

    for (model, vals), color in zip(items, colors):
        x = vals["miu_l"]
        y = vals["miu_r"]
        ax.scatter(x, y, s=90, color=color, edgecolors="white", linewidths=0.9, alpha=0.95, zorder=3)

    # Annotate with slight x-offset to improve readability
    x_span = max(x_values) - min(x_values) if len(x_values) > 1 else 1.0
    x_offset = 0.012 * x_span
    for (model, vals) in items:
        ax.text(vals["miu_l"] + x_offset, vals["miu_r"], model, fontsize=10, va="center", ha="left", alpha=0.9, zorder=4)

    ax.set_xlabel("L: Expected output tokens")
    ax.set_ylabel("r: Expected reward")
    ax.set_title("Model Efficiency–Reward Frontier (L vs r)")

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_dir = os.path.join("outputs", "nl_graph")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model_efficiency_reward_frontier.pdf")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)



data_path_format = 'data/local_records/mmlu-pro/{}_test_result.jsonl'
models = ['gpt-4', 'gpt-35-turbo', 'gpt-4o-mini', 'qwen-max', 'gpt-4o', 'deepseek-v3', 'deepseek-r1', 'o1-mini', 'o1', 'claude-4-0', 'claude-3-7', 'o3-mini']
plot_data = {}

for model in models:
    data = [json.loads(item) for item in open(data_path_format.format(model)).readlines()]

    miu_l = np.mean([item['output_tokens'] for item in data])
    miu_r = np.mean([item['score'] for item in data]) * 7

    plot_data[model] = {
        "miu_l": miu_l,
        "miu_r": miu_r
    }


if __name__ == '__main__':
    plot_scatter()
