import json
from PIL.Image import SAVE
from dill import load
import networkx as nx
from openai import AzureOpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from collections import deque

from src.utils.data import get_gsm8k
import os
from dotenv import load_dotenv
load_dotenv()


        

# def run_single_task(client, task):
#     prompt = translate(task["G"], task["q"], 'cot')
#     response = client.messages.create(
#                         model="us.anthropic.claude-sonnet-4-20250514-v1:0",
#                         messages=[{
#                             "role": "user",
#                             "content": prompt,
#                             }],
#                         max_tokens=10000
#                     )

#     llm_answer, input_tokens, output_tokens = response.content[0].text, response.usage.input_tokens, response.usage.output_tokens
#     score = evaluate(llm_answer.lower(), task["G"], task["q"], task["correct_answer"])
#     return {
#         "id": task["id"],
#         "score": score,
#         "input_tokens": input_tokens,
#         "output_tokens": output_tokens,
#         "llm_answer": llm_answer
#     }

def prepare_openai_data(data, model):
    batch_input = []
    for item in data:
        batch_input.append(
            json.dumps({
                "custom_id": f"request-{len(batch_input)+1}", 
                "method": "POST", 
                "url": "/v1/chat/completions", 
                "body": {
                    "model": model, 
                    "messages": item,
                    "max_tokens": 100000
                    }
                    }) + '\n'

        )
    with open(f'data/gsm8k/batch_input_{model}.jsonl', 'w') as f:
        f.writelines(batch_input)


if __name__ == '__main__':
    # data = get_gsm8k()[:16]
    # model = 'gpt-4.1'
    # prepare_openai_data(data, model)
    client = AzureOpenAI(
                azure_endpoint=os.getenv("OPENAI_ENDPOINT"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION")
            )
    # batch_input_file = client.files.create(
    #     file=open(f"data/gsm8k/batch_input_{model}.jsonl", "rb"),
    #     purpose="batch"
    #     )
    # batch_input_file_id = batch_input_file.id
    # batch_process = client.batches.create(
    #     input_file_id=batch_input_file_id,
    #     endpoint="/v1/chat/completions",
    #     completion_window="24h",
    #     metadata={
    #         "description": "nightly eval job"
    # })

    # print(batch_process)

    batch = client.batches.retrieve("batch_1d02f680-75ca-4c0c-ac50-4ba14cdf90a1")
    print(batch)

    # prepare_openai_data


    # results = json.load(open(SAVE_PATH)) if os.path.exists(SAVE_PATH) else []
    # idx_done = [item["id"] for item in results]
    
    
    # total_tasks = len(tasks)
    # print(f"Total tasks: {total_tasks}")
    # pending = [(i, t) for i, t in enumerate(tasks) if i not in idx_done]
    # print(f"Already done: {len(idx_done)}; To run: {len(pending)}")
    # if pending:
    #     max_workers = 4
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         running_ids = set()
    #         completed_count = 0
    #         future_to_idx = {}
    #         pending_deque = deque(pending)

    #         # Preload up to max_workers tasks
    #         preload = min(max_workers, len(pending_deque))
    #         for _ in range(preload):
    #             i, t = pending_deque.popleft()
    #             print(f"Starting task {i} (max_workers={max_workers})")
    #             future = executor.submit(run_single_task, client, t)
    #             future_to_idx[future] = i
    #             running_ids.add(i)
    #         print(f"Currently running: {sorted(running_ids)}; Not started: {len(pending_deque)}")

    #         while future_to_idx:
    #             done, _ = wait(list(future_to_idx.keys()), return_when=FIRST_COMPLETED)
    #             for future in done:
    #                 i = future_to_idx.pop(future)
    #                 try:
    #                     result = future.result()
    #                 except Exception as exc:
    #                     result = {
    #                         "id": i,
    #                         "score": 0,
    #                         "input_tokens": 0,
    #                         "output_tokens": 0,
    #                         "llm_answer": f"error: {exc}"
    #                     }

    #                 running_ids.discard(i)
    #                 completed_count += 1
    #                 remaining_total = total_tasks - (len(idx_done) + completed_count)
    #                 print(f"Completed task {i}; Remaining (total): {remaining_total}")
    #                 results.append(result)
    #                 results = sorted(results, key=lambda x: x["id"])
    #                 json.dump(results, open(SAVE_PATH, 'w'), indent=2)


    #                 if pending_deque:
    #                     next_i, next_t = pending_deque.popleft()
    #                     print(f"Starting task {next_i} (max_workers={max_workers})")
    #                     next_future = executor.submit(run_single_task, client, next_t)
    #                     future_to_idx[next_future] = next_i
    #                     running_ids.add(next_i)
    #             if future_to_idx:
    #                 print(f"Currently running: {sorted(running_ids)}; Not started: {len(pending_deque)}")
