import csv
import json
from datasets import load_dataset
import numpy as np
import random




def get_gsm8k(split='test'):

    from src.utils.gsm8k_eval import GSM8kEvaluator
    
    if split not in ['train', 'test']:
        raise ValueError(f"split {split} not maintained in this dataset")

    dataset = load_dataset("json", data_files=f"data/gsm8k/test.json", split='train')


    questions = list(dataset['question'])
    answers = list(dataset['answer'])
    num_samples = len(questions)

    evaluation_data = []


    for i in range(num_samples):

        user_prompt = questions[i]
        
        ground_truth = answers[i]
        ground_truth = GSM8kEvaluator.extract_answer(ground_truth)

        evaluation_data.append( (user_prompt, ground_truth) )
    
    return evaluation_data