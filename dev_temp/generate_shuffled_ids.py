import os
import random
import json

a = list(range(0, 2000)) * 5
random.shuffle(a)
json.dump(a, open('data/task_ids_shuffled.json', 'w'))