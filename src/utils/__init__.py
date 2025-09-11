import json
from src.utils.logger import Logger
def load_jsonl(path):
    return [json.loads(line) for line in open(path)]