import json
from pathlib import Path
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[1]
jsonl_path = project_root / 'data/local_records/nlgraph_new/gpt-5-high_test_result.jsonl'


with open(jsonl_path, 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


data = [rec.get('output_tokens') for rec in records]
data = [v for v in data if _is_number(v)]


plt.figure(figsize=(8, 5))
plt.hist(data, bins='auto', edgecolor='black', alpha=0.7)
plt.title('Frequency distribution of output_tokens')
plt.xlabel('output_tokens')
plt.ylabel('Frequency')
plt.tight_layout()


output_png = Path(__file__).resolve().parent / 'output_tokens_hist.png'
plt.savefig(output_png, dpi=150)
plt.show()

