import random
import json


a = list(range(1000))

random.shuffle(a)

json.dump(a, open('shuffle.json', 'w'))
