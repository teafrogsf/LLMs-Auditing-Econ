import json
from pathlib import Path
import sys
import math 


def the_func(t):
    return 0.742 * t #- (t**(1-0.3)*math.log(t) + t**(0.6))
# Ensure project root is on sys.path so that `src` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.ana import get_result_all_ours


ROOT_DIR = Path('outputs/nl_graph/aba_T_yuhan1')
user_utility = {}
the_utility = {}
for i in range(1000000, 2010000, 10000):
    data_dir = ROOT_DIR / f'T{i}'
    result = get_result_all_ours(0, data_dir)
    user_utility[i] = result['0']['user_utility']
    the_utility[i] = the_func(i)
    

# Plot utilities vs T (Number of tasks) with science-paper aesthetics
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Style and aesthetics
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

T_values = sorted(user_utility.keys())
user_values = [user_utility[T] for T in T_values]

the_values = [the_utility[T] for T in T_values]

def _format_millions(x, pos):
    # Show ticks in millions for readability
    return f'{x * 1e-6:.2f}M'

fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=300)

try:
    colors = mpl.colormaps['tab10'].colors
except Exception:
    colors = plt.get_cmap('tab10').colors  # fallback for older matplotlib

ax.plot(T_values, user_values,
        color=colors[0],
        linewidth=2.2,
        marker='o',
        markersize=3.5,
        alpha=0.95,
        label='Actual user utility')

ax.plot(T_values, the_values,
        color=colors[1],
        linewidth=2.0,
        linestyle='--',
        alpha=0.9,
        label='Second-best user utility')

ax.set_xlabel('T (Number of queries)')
ax.set_ylabel('Utility')
ax.tick_params(axis='y', labelcolor='black')

ax.xaxis.set_major_formatter(FuncFormatter(_format_millions))

ax.legend(loc='upper left')

fig.tight_layout()

# Ensure output directory exists and save
ROOT_DIR.mkdir(parents=True, exist_ok=True)
for ext in ['png', 'pdf']:
    fig.savefig(ROOT_DIR / f'utility_vs_T_all_ours_lb.{ext}', bbox_inches='tight')

plt.close(fig)
