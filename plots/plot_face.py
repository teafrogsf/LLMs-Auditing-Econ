import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
CHOICES = ['honest', 'ours', 'worst', 'random', 'h1w2', 'w1h2']
ROOT = Path('outputs/toy_game/default')

def plot(data, save_path):
    """
    Create a bar chart from the data dictionary
    
    Args:
        data: Dictionary with keys as strategy names and values as utility values
        save_path: Path to save the plot
    """
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Extract keys and values
    strategies = list(data.keys())
    utilities = list(data.values())
    
    # Create bar chart
    bars = plt.bar(strategies, utilities, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1.2)
    
    # Customize the plot
    plt.xlabel('Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Total Provider Utility', fontsize=12, fontweight='bold')
    plt.title('Provider Utility by Strategy', fontsize=14, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, utilities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(utilities)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved to: {save_path}")

results_dir = [ ROOT / item for item in os.listdir(ROOT) if item.startswith('honest')]
for stragety in CHOICES:
    results_dir_temp = [item for item in results_dir if item.name.split('-')[-2]==stragety]
    

    
    data = {}
    for sc in results_dir_temp:
        sc_name = sc.name
        print(sc_name)
        result_path = sc / 'result.json'
        result = json.load(open(result_path))
        # print(result)
        data[sc_name] = result['providers'][1]['total_provider_utility']
    
    plot(data, save_path=f'outputs/toy_game/default/figs/provider2/{stragety}.pdf')
