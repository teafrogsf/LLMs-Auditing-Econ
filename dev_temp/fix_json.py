import json

data = json.load(open('data/max_flow_graphs.json'))

for k in data:
    data[k]["id"] = k



data_items_sorted = sorted(list(data.values()), key=lambda x: x["id"])

data_new = []
for item in data_items_sorted:
    new_item = {
        "id": item["id"],
        "graph": item["graph"],
        "source": item["source"],
        "target": item["target"]
    }
    data_new.append(new_item)
json.dump(data_new, open('data/max_flow_graphs_local.json', 'w') ,indent=2)