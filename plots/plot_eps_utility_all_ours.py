import json
import os

from pathlib import Path
import sys
import argparse

# Ensure project root is on sys.path so that `src` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ana import get_result_all_ours

paser = argparse.ArgumentParser()
paser.add_argument('--output-dir', type=str, default='outputs/nl_graph/aba_eps_finnal')
args = paser.parse_args()

ROOT_DIR = Path(args.output_dir)
user_utility = {}
provider_utility = {}
for i in range(15, 40, 1):
    data_dir = ROOT_DIR / f'eps{i}'
    print(data_dir)
    result = get_result_all_ours(0, data_dir)
    user_utility[i] = result['0']['user_utility']
    provider_utility[i] = result['0']['provider_utility']

json.dump(provider_utility, open(ROOT_DIR/ 'provider_utility.json', 'w'), indent=2)


# Plot utilities vs ε with science-paper aesthetics and dual y-axes
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        pass

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
})

eps_keys = sorted(user_utility.keys())
eps_values = [k / 100.0 for k in eps_keys]
user_values = [user_utility[k] for k in eps_keys]
provider_values = [provider_utility[k] for k in eps_keys]

fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=300)
ax2 = ax.twinx()

try:
    colors = mpl.colormaps['tab10'].colors
except Exception:
    colors = plt.get_cmap('tab10').colors

user_line, = ax.plot(eps_values, user_values,
                     label='User Utility',
                     color=colors[0],
                     linewidth=2.2,
                     marker='o',
                     markersize=3.5,
                     alpha=0.95)

provider_line, = ax2.plot(eps_values, provider_values,
                          label='Provider Utility',
                          color=colors[3],
                          linewidth=2.2,
                          marker='s',
                          markersize=3.5,
                          alpha=0.95)

ax.set_xlabel('ε')
ax.set_ylabel('User Utility', color=colors[0])
ax2.set_ylabel('Provider Utility', color=colors[3])
ax.tick_params(axis='y', labelcolor=colors[0])
ax2.tick_params(axis='y', labelcolor=colors[3])
ax2.spines['right'].set_visible(True)

# Major ticks every 0.05 (e.g., 0.20, 0.25, ...)
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

lines = [user_line, provider_line]
labels = [line.get_label() for line in lines]
ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.9, 1.2), borderaxespad=0.2)

fig.tight_layout()

ROOT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ['png', 'pdf']:
    fig.savefig(ROOT_DIR / f'utility_vs_eps_all_ours.{ext}', bbox_inches='tight')

plt.close(fig)
