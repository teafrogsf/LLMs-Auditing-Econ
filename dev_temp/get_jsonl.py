import json
data_path = 'data/claude-4-0_test_result_new.json'

data = json.load(open(data_path))

with open(data_path.replace('.json', '.jsonl'), 'w') as f:
    for item in data:
        item.pop('llm_answer')
        f.write(json.dumps(item) + '\n')