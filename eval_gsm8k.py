import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any

from src.utils.data import get_gsm8k
from src.model import ExampleLLM, MODEL_PRICING
from src.utils.gsm8k_eval import GSM8kEvaluator


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def build_prompt(question: str) -> str:
    return (
        "You are a helpful math tutor. Solve the following problem step by step. "
        "Show your reasoning briefly and then give the final numeric answer on a new line in the exact format: #### <number>.\n\n"
        f"Problem: {question}\n\n"
        "Answer format:\n"
        "- Reasoning...\n"
        "- Final line: #### 1234\n"
    )


def safe_to_float(text: str) -> Tuple[bool, float]:
    try:
        return True, float(text)
    except Exception:
        return False, 0.0


def evaluate_model_on_gsm8k(model_name: str, flush_every: int = 20) -> None:
    tasks: List[Tuple[str, str]] = get_gsm8k()
    result_path = f"data/local_records/gsm8k/{model_name}.json"
    ensure_dir(result_path)

    try:
        with open(result_path, "r") as f:
            model_results: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        model_results = []
    except json.JSONDecodeError:
        model_results = []

    start_index = len(model_results)
    llm = ExampleLLM(model_name)

    total_correct = sum(r.get("score", 0) for r in model_results)
    total_prompt_tokens = sum(r.get("input_tokens", 0) for r in model_results)
    total_completion_tokens = sum(r.get("output_tokens", 0) for r in model_results)

    for idx, (question, gt_answer_text) in enumerate(tasks[start_index:], start=start_index):
        prompt = build_prompt(question)
        response_text, prompt_tokens, completion_tokens = llm.call_llm(prompt)
        
        parsed_model_answer = GSM8kEvaluator.extract_answer(response_text)
        parsed_gt_answer = gt_answer_text  # already extracted in data loader

        is_valid_model_ans, model_ans_float = safe_to_float(parsed_model_answer)
        is_valid_gt_ans, gt_ans_float = safe_to_float(parsed_gt_answer)

        score = 0
        if is_valid_model_ans and is_valid_gt_ans:
            score = 1 if model_ans_float == gt_ans_float else 0

        record = {
            "id": idx + 1,
            "score": score,
            "input_tokens": int(prompt_tokens),
            "output_tokens": int(completion_tokens),
            "output": parsed_model_answer,
            "raw_response": response_text,
            "gt_answer": parsed_gt_answer,
        }
      
        model_results.append(record)

        total_correct += score
        total_prompt_tokens += int(prompt_tokens)
        total_completion_tokens += int(completion_tokens)

        if (len(model_results) % flush_every) == 0:
            with open(result_path, "w") as f:
                json.dump(model_results, f, ensure_ascii=False, indent=2)
            print(f"[{model_name}] Progress: {len(model_results)}/{len(tasks)} saved.")

    # Final save
    with open(result_path, "w") as f:
        json.dump(model_results, f, ensure_ascii=False, indent=2)
    # No summary output requested
    return None


def atomic_write_json(path: str, data: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def prepare_model_state(model_name: str) -> Dict[str, Any]:
    result_path = f"data/local_records/gsm8k/{model_name}.json"
    ensure_dir(result_path)
    try:
        with open(result_path, "r") as f:
            existing: List[Dict[str, Any]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []
    completed_ids = {int(r.get("id", -1)) for r in existing if isinstance(r, dict)}
    state = {
        "model": model_name,
        "llm": ExampleLLM(model_name),
        "results_path": result_path,
        "results": existing,
        "completed_ids": completed_ids,
        "lock": threading.Lock(),
    }
    return state


def process_one(state: Dict[str, Any], task_idx: int, question: str, gt_answer_text: str) -> None:
    model_name = state["model"]
    llm: ExampleLLM = state["llm"]
    result_id = task_idx + 1

    if result_id in state["completed_ids"]:
        return

    prompt = build_prompt(question)

    backoffs = [0, 1, 2, 4]
    error_msg = None
    response_text = None
    prompt_tokens = 0
    completion_tokens = 0
    for delay in backoffs:
        if delay:
            time.sleep(delay)
        try:
            response_text, prompt_tokens, completion_tokens = llm.call_llm(prompt)
            error_msg = None
            break
        except Exception as e:
            error_msg = str(e)
            continue

    parsed_model_answer = GSM8kEvaluator.extract_answer(response_text) if error_msg is None else "[error]"
    parsed_gt_answer = gt_answer_text

    is_valid_model_ans, model_ans_float = safe_to_float(parsed_model_answer)
    is_valid_gt_ans, gt_ans_float = safe_to_float(parsed_gt_answer)
    score = 0
    if error_msg is None and is_valid_model_ans and is_valid_gt_ans:
        score = 1 if model_ans_float == gt_ans_float else 0

    record = {
        "id": result_id,
        "score": score,
        "input_tokens": int(prompt_tokens) if error_msg is None else 0,
        "output_tokens": int(completion_tokens) if error_msg is None else 0,
        "output": parsed_model_answer,
        "raw_response": response_text if error_msg is None else None,
        "gt_answer": parsed_gt_answer,
        "error": error_msg,
    }

    with state["lock"]:
        if result_id not in state["completed_ids"]:
            state["results"].append(record)
            state["completed_ids"].add(result_id)
            atomic_write_json(state["results_path"], state["results"])
            print(f"[{model_name}] saved id={result_id} (total {len(state['results'])})")


def evaluate_all_models_concurrent(models: List[str], max_workers: int = 16) -> None:
    tasks: List[Tuple[str, str]] = get_gsm8k()
    states: Dict[str, Dict[str, Any]] = {m: prepare_model_state(m) for m in models}

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_name, state in states.items():
            for idx, (question, gt_answer_text) in enumerate(tasks):
                result_id = idx + 1
                if result_id in state["completed_ids"]:
                    continue
                futures.append(executor.submit(process_one, state, idx, question, gt_answer_text))

        for _ in as_completed(futures):
            pass


if __name__ == "__main__":
    model_list = ['gpt-4o']
    max_workers = int(os.getenv("EVAL_MAX_WORKERS", "16"))
    try:
        evaluate_all_models_concurrent(model_list, max_workers=max_workers)
    except Exception as e:
        print(f"Error during concurrent evaluation: {e}")