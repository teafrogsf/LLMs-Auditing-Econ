import json
import os
import numpy as np
from pathlib import Path

DATA_PATH = Path('data/local_records/nlgraph_new')

data_path = [DATA_PATH / item for item in os.listdir(DATA_PATH) if item.endswith('.jsonl')]
data = []
for dp in data_path:
    data.append(np.array([json.loads(line)['input_tokens'] for line in open(dp).readlines()]))
    output_tokens = np.array([json.loads(line)['output_tokens'] for line in open(dp).readlines()])
    scores = np.array([json.loads(line)['score'] for line in open(dp).readlines()])
    input_tokens = np.array([json.loads(line)['input_tokens'] for line in open(dp).readlines()])
    print(str(dp).replace('.jsonl', '.npz'))
    # exit()
    np.savez(str(dp).replace('.jsonl', ''),input_tokens=input_tokens, output_tokens=output_tokens, scores=scores)
# data = np.vstack(data)
# data = np.mean(data, axis=0).astype(int)
# print(data[:5])
# np.savez('data/local_records/nlgraph_new/input_tokens.npz', data=data)