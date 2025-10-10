import json
import os
import numpy as np
from pathlib import Path
import yaml

record_path = Path('data/local_records/nlgraph_new')
model_info = {}


MODEL_PRICING = {
    "claude-4-0": {"input": 3/1_000_000, "output": 15/1_000_000},
    "gpt-4o": {"input": 2.5/1_000_000, "output": 10/1_000_000},
    "gpt-4": {"input": 30/1_000_000, "output": 60/1_000_000},
    "gpt-4.1": {"input": 3/1_000_000, "output": 12/1_000_000},
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.6/1_000_000},
    "o1-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "o3-mini": {"input": 1.1/1_000_000, "output": 4.4/1_000_000},
    "gpt-35-turbo": {"input": 0.5/1_000_000, "output": 1.5/1_000_000},
    "qwen-max": {"input": 1.6/1_000_000, "output": 6.4/1_000_000},
    "deepseek-v3": {"input": 0.07/1_000_000, "output": 1.10/1_000_000},
    "deepseek-r1": {"input": 0.14/1_000_000, "output": 2.19/1_000_000},
    "o1": {"input": 15/1_000_000, "output": 60/1_000_000},
    "gpt-5-high": {"input": 1.25/1_000_000, "output": 10.0/1_000_000},
    "gpt-5-low": {"input": 1.25/1_000_000, "output": 10.0/1_000_000},
    "gpt-5-medium": {"input": 1.25/1_000_000, "output": 10.0/1_000_000},
    "o4-mini": {"input": 4/1_000_000, "output": 16/1_000_000}
}

model_result_path = [record_path / item for item in os.listdir(record_path) if  item.endswith('.jsonl')]
L = 0


for path in model_result_path:
    model_name = path.name.split('_')[0]
    results = [json.loads(item) for item in open(path).readlines()]
    assert len(results) == 2000
    
    scores = [item['score'] for item in results]
    output_tokens = [item['output_tokens'] for item in results]


    score_mu = float(np.mean(scores))
    output_tokens_mu = float(np.mean(output_tokens))
    max_output_tokens = max(output_tokens)
    input_token_price = MODEL_PRICING[model_name]['input']
    output_token_price = MODEL_PRICING[model_name]['output']

    price_mu = output_tokens_mu * output_token_price
    # user_utility_mu = 5 * score_mu - price_mu
    model_info[model_name] = {
        'score_mu': score_mu,
        'output_tokens_mu': output_tokens_mu,
        'input_token_price': input_token_price,
        'output_token_price': output_token_price,
        'max_output_tokens': max_output_tokens,
        'price_mu': price_mu,
    }


# print(sorted([model for model in model_info], key=lambda x: model_info[x]['user_utility_mu'], reverse=True))
print(sorted([model for model in model_info], key=lambda x: model_info[x]['price_mu'], reverse=True))
print(sorted([model for model in model_info], key=lambda x: model_info[x]['output_token_price'], reverse=True))



with open('config/nl_graph/model_config.yaml', 'w') as f:
    yaml.dump(model_info, f)


