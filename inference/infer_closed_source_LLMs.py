import json
from transformers import pipeline
import torch
import argparse
from tqdm import tqdm
import os
import sys
sys.path.append('..')
from llm_client import ExampleLLM


def custom_format(template, variables):
    for key, value in variables.items():
        if isinstance(value, str):
            value = '`'+ value + '`'
            exec(f'{key} = {repr(value)}')
        else:
            exec(f'{key} = {value}')
    
    result = eval(f'f"""{template}"""')
    return result

def main(args):
    with open(f'../data/test/{args.dataset}.json', 'r') as fp:
        datasets = json.load(fp)
    os.makedirs(f"../infer_data/{args.model_path}/{args.method}", exist_ok=True)
    try:
        with open(f'../infer_data/{args.model_path}/{args.method}/{args.dataset}.json', 'r') as fp:
            infer_data = json.load(fp)
    except:
        infer_data = []
    
    # 自定义模型列表
    supported_models = ['gpt-4o-mini', 'gpt-4o', 'gpt-4', 'gpt-35-turbo-0125-60ktpm', 
                       'o1-mini-1mtpm', 'o1', 'o3-mini-1mtpm', 'qwen-max', 
                       'deepseek-r1', 'deepseek-v3', 'deepseek-chat', 'deepseek-reasoner']
    
    if args.model_path not in supported_models:
        raise ValueError(f"Unsupported model: {args.model_path}. Supported models: {supported_models}")
    
    custom_llm = ExampleLLM(args.model_path)

    data_length = len(infer_data)//3
    if datasets[data_length : ] == []:
        raise ValueError("You've finished generating!")

    for data in tqdm(datasets[data_length : ]):
        for query in data["query"]:
            formatted_query = custom_format(query, data["variables"])
            if args.method == 'raw':
                prompt = f"Solve the question below.\n\n{formatted_query}\n\nProvide the final result like this:\n```llm_result\nxxx\n```\n"

            elif args.method == 'pot':
                prompt = f"Write a Python code to solve the question below.\n\n{formatted_query}\n\nJust provide the code, no explainations."

            elif args.method == 'cot':
                prompt = f"Solve the question below.\n\n{formatted_query}\n\nUse chain-of-thought to solve it, and make sure to provide the final result like this:\n```llm_result\nxxx\n```\n"

            else:
                raise ValueError("method not found")
            
            try:
                each_result, _, _ = custom_llm.call_llm(prompt)
            except:
                try:
                    each_result, _, _ = custom_llm.call_llm(prompt)
                except:
                    each_result = ''
            if args.method == 'cot' or args.method == 'raw':
                each_result = each_result[each_result.find('```llm_result') + len('```llm_result'):]
                each_result = each_result[:each_result.find('```')]
                each_result = each_result.strip("\n")
            elif args.method == 'pot':
                if "```python" in each_result:
                    each_result = each_result[each_result.find('```python') + len('```python'):]
                    each_result = each_result[:each_result.find('```')]
                each_result = each_result.strip()
            infer_data.append({"prompt": prompt, "solution": data["solution"], "response": each_result, "variables": data["variables"]})

    
        with open(f'../infer_data/{args.model_path}/{args.method}/{args.dataset}.json', 'w') as fp:
            json.dump(infer_data, fp, ensure_ascii=False, indent = 4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='dataset name')
    parser.add_argument('--temperature', default=0.8, type=float, help='inference temperature')
    parser.add_argument('--top_p', default=0.95, type=float, help='inference top_p')
    parser.add_argument('--model_path', required=True, type=str, help='model path')
    parser.add_argument('--tensor_parallel_size', default=1, type=int, help='gpu numbers')
    parser.add_argument('--method', default='raw', type=str, help='prompt engineering technique')
    args = parser.parse_args()
    while(1):
        try:
            main(args)
            break
        except json.decoder.JSONDecodeError:
            continue
        except Exception as e:
            print(f"Error: {e}")
            break
