import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

import datasets
import random

from src.model import ExampleLLM, MODEL_PRICING


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def atomic_write_json(path: str, data: List[Dict[str, Any]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def form_options(options: List[str]) -> str:
    option_str = "Options are:\n"
    opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for opt, letter in zip(options, opts):
        option_str += f"({letter}): {opt}\n"
    return option_str


def build_prompts_from_validation(dataset: datasets.DatasetDict) -> Dict[str, str]:
    categories = set(dataset["validation"]["category"]) if "validation" in dataset else set()
    prompts: Dict[str, str] = {c: "" for c in categories}
    if "validation" not in dataset:
        return prompts

    for entry in dataset["validation"]:
        category = entry["category"]
        q = entry["question"]
        options = entry["options"]
        cot_content = entry.get("cot_content", "")
        prompts[category] += (
            "Q: " + q + "\n" + form_options(list(options)) + "\n" + cot_content + "\n\n"
        )
    return prompts


def get_prediction(output: Optional[str]) -> Optional[str]:
    if not output:
        return None
    import re
    pattern = re.compile(r"answer is \(?([ABCDEFGHIJ])\)?", re.IGNORECASE)
    match = pattern.search(output)
    if match:
        return match.group(1).upper()
    return None


def prepare_model_state(model_name: str) -> Dict[str, Any]:
    result_path = f"data/local_records/mmlu-pro/{model_name}.json"
    ensure_dir(result_path)
    try:
        with open(result_path, "r") as f:
            existing: List[Dict[str, Any]] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    completed_qids = {int(r.get("question_id", -1)) for r in existing if isinstance(r, dict)}
    state = {
        "model": model_name,
        "llm": ExampleLLM(model_name),
        "results_path": result_path,
        "results": existing,
        "completed_qids": completed_qids,
        "lock": threading.Lock(),
    }
    return state


def run_one_question_with_llm(llm: ExampleLLM, prefix_prompt: str, entry: Dict[str, Any]) -> Tuple[str, int, int]:
    question = entry["question"]
    options = entry["options"]
    user_query = prefix_prompt + "Q: " + question + "\n" + form_options(list(options)) + "\n"
    return llm.call_llm(user_query)


def process_one(state: Dict[str, Any], entry: Dict[str, Any], prefix_prompt: str) -> None:
    model_name = state["model"]
    llm: ExampleLLM = state["llm"]

    question_id = int(entry["question_id"]) if "question_id" in entry else int(entry.get("id", -1))
    if question_id in state["completed_qids"]:
        return

    # Print progress: model and question id
    print(f"Running model={model_name}, question_id={question_id}")

    error_msg = None
    response_text = None
    prompt_tokens = 0
    completion_tokens = 0

    backoffs = [0, 1, 2, 4]
    for delay in backoffs:
        if delay:
            import time
            time.sleep(delay)
        try:
            response_text, prompt_tokens, completion_tokens = run_one_question_with_llm(llm, prefix_prompt, entry)
            error_msg = None
            break
        except Exception as e:
            error_msg = str(e)
            continue

    # Parse prediction and compute score
    predicted_letter = get_prediction(response_text) if error_msg is None else None
    gt_letter: Optional[str] = None
    gt_answer = entry.get("answer")
    if isinstance(gt_answer, str) and gt_answer:
        gt_letter = gt_answer.strip().upper()
    elif "answer_index" in entry and isinstance(entry["answer_index"], int):
        opts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        idx = entry["answer_index"]
        if 0 <= idx < len(opts):
            gt_letter = opts[idx]

    score = 1 if (predicted_letter is not None and gt_letter is not None and predicted_letter == gt_letter) else 0

    record: Dict[str, Any] = {
        "question_id": question_id,
        "score": score,
        "input_tokens": int(prompt_tokens) if error_msg is None else 0,
        "output_tokens": int(completion_tokens) if error_msg is None else 0,
        "llm_raw_answer": response_text if error_msg is None else None,
    }

    with state["lock"]:
        if question_id not in state["completed_qids"]:
            state["results"].append(record)
            state["completed_qids"].add(question_id)
            atomic_write_json(state["results_path"], state["results"])


def evaluate_all_models_concurrent(models: List[str], max_workers: int = 16) -> None:
    # Load local dataset from data/mmlu-pro
    dataset = datasets.load_dataset(
        path="data/mmlu-pro",
        name="default",
    )

    # Build category-specific few-shot prefixes from validation split
    prompts = build_prompts_from_validation(dataset)

    # Deterministic shuffle and subset for test split
    # Use a fixed seed so every model sees the same 2000 examples
    rng = random.Random(20250920)
    test_list: List[Dict[str, Any]] = [ex for ex in dataset["test"]]
    rng.shuffle(test_list)
    test_list = test_list[:2000]

    # Prepare per-model state
    states: Dict[str, Dict[str, Any]] = {m: prepare_model_state(m) for m in models}

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Interleave submissions across models to ensure cross-model concurrency
        model_items: List[Tuple[str, Dict[str, Any]]] = list(states.items())
        for entry in test_list:
            qid = int(entry["question_id"]) if "question_id" in entry else int(entry.get("id", -1))
            category = entry.get("category")
            prefix = prompts.get(category, "")
            for model_name, state in model_items:
                if qid in state["completed_qids"]:
                    continue
                futures.append(executor.submit(process_one, state, entry, prefix))

        for _ in as_completed(futures):
            pass


if __name__ == "__main__":
    model_list = list(MODEL_PRICING.keys())
    max_workers = int(os.getenv("EVAL_MAX_WORKERS", "64"))
    try:
        evaluate_all_models_concurrent(model_list, max_workers=max_workers)
    except Exception as e:
        print(f"Error during MMLU-Pro evaluation: {e}")


