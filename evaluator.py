import json
from PIL.Image import SAVE
import networkx as nx
import anthropic
import os
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from collections import deque
from src.nl_graph.max_flow_solver import translate, evaluate

DATA_PATH = 'data/max_flow_graphs_local.json'
SAVE_PATH = 'data/claude-4-0_test_result.json'



def load_tasks(path=DATA_PATH):
    data = json.load(open(path))
    tasks = []
    for idx, graph_info in enumerate(data):
        graph_data = json.loads(graph_info['graph'])
        G = nx.node_link_graph(graph_data)
        source, target = graph_info['source'], graph_info['target']
        q = (source, target)
        correct_answer = nx.maximum_flow_value(G, source, target, capacity='capacity')
   

        tasks.append({
            "id": idx,
            "G": G,
            "q": q,
            "correct_answer": correct_answer}
        )
    return tasks
        

def run_single_task(client, task):
    prompt = translate(task["G"], task["q"], 'cot')
    response = client.messages.create(
                        model="us.anthropic.claude-sonnet-4-20250514-v1:0",
                        messages=[{
                            "role": "user",
                            "content": prompt,
                            }],
                        max_tokens=10000
                    )

    llm_answer, input_tokens, output_tokens = response.content[0].text, response.usage.input_tokens, response.usage.output_tokens
    score = evaluate(llm_answer.lower(), task["G"], task["q"], task["correct_answer"])
    return {
        "id": task["id"],
        "score": score,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "llm_answer": llm_answer
    }

if __name__ == '__main__':
    client = anthropic.AnthropicBedrock(
                    aws_region=os.getenv("AWS_REGION"),
                    aws_access_key=os.getenv("AWS_ACCESS_KEY"),
                    aws_secret_key=os.getenv("AWS_SECRET_KEY"),
                    http_client=httpx.Client(proxy=os.getenv("PROXY"))
                )
    tasks = load_tasks()
    
    results = json.load(open(SAVE_PATH)) if os.path.exists(SAVE_PATH) else []
    idx_done = [item["id"] for item in results]
    
    
    total_tasks = len(tasks)
    print(f"Total tasks: {total_tasks}")
    pending = [(i, t) for i, t in enumerate(tasks) if i not in idx_done]
    print(f"Already done: {len(idx_done)}; To run: {len(pending)}")
    if pending:
        max_workers = 4
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            running_ids = set()
            completed_count = 0
            future_to_idx = {}
            pending_deque = deque(pending)

            # Preload up to max_workers tasks
            preload = min(max_workers, len(pending_deque))
            for _ in range(preload):
                i, t = pending_deque.popleft()
                print(f"Starting task {i} (max_workers={max_workers})")
                future = executor.submit(run_single_task, client, t)
                future_to_idx[future] = i
                running_ids.add(i)
            print(f"Currently running: {sorted(running_ids)}; Not started: {len(pending_deque)}")

            while future_to_idx:
                done, _ = wait(list(future_to_idx.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    i = future_to_idx.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "id": i,
                            "score": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "llm_answer": f"error: {exc}"
                        }

                    running_ids.discard(i)
                    completed_count += 1
                    remaining_total = total_tasks - (len(idx_done) + completed_count)
                    print(f"Completed task {i}; Remaining (total): {remaining_total}")
                    results.append(result)
                    results = sorted(results, key=lambda x: x["id"])
                    json.dump(results, open(SAVE_PATH, 'w'), indent=2)


                    if pending_deque:
                        next_i, next_t = pending_deque.popleft()
                        print(f"Starting task {next_i} (max_workers={max_workers})")
                        next_future = executor.submit(run_single_task, client, next_t)
                        future_to_idx[next_future] = next_i
                        running_ids.add(next_i)
                if future_to_idx:
                    print(f"Currently running: {sorted(running_ids)}; Not started: {len(pending_deque)}")
