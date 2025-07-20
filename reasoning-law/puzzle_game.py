import random
from typing import List, Tuple
from tqdm import tqdm

import sys
sys.path.append("..")
from llm_client import ExampleLLM

OPERATORS = ['+', '-', '*', '/']


def _generate_postfix_expression(nums: List[int]) -> List[str]:
    """在给定数字列表的情况下随机生成一个合法的后缀（RPN）表达式。

    生成规则：
    1. 对于 n 个数字，需要 n-1 个二元运算符。
    2. 维护栈深度 stack_size，初始为 0；
       - 每放入一个操作数，stack_size += 1
       - 每放入一个运算符，stack_size -= 1（因为两数合一），要求放入运算符前 stack_size >= 2。
    3. 最终遍历结束时 stack_size 必须为 1。
    """
    operands = nums.copy()
    random.shuffle(operands)  # 随机打乱数字顺序
    operators_needed = len(nums) - 1
    tokens: List[str] = []
    stack_size = 0

    while operands or operators_needed:
        if (stack_size < 2) or (operands and random.choice([True, False])):
            # 必须放入操作数，或随机决定继续放操作数
            val = operands.pop()
            tokens.append(str(val))
            stack_size += 1
        else:
            # 放入运算符
            op = random.choice(OPERATORS)
            tokens.append(op)
            operators_needed -= 1
            stack_size -= 1

    return tokens


def _eval_postfix(tokens: List[str]) -> float:
    """计算后缀表达式的值。"""
    stack: List[float] = []
    try:
        for tok in tokens:
            if tok in OPERATORS:
                b = stack.pop()
                a = stack.pop()
                if tok == '+':
                    stack.append(a + b)
                elif tok == '-':
                    stack.append(a - b)
                elif tok == '*':
                    stack.append(a * b)
                elif tok == '/':
                    stack.append(a / b)
            else:
                stack.append(float(tok))
        return stack[0] if stack else 0.0
    except ZeroDivisionError:
        # 向上传递特殊值，供调用方判断
        raise ValueError("Division by zero in postfix eval")


def generate_puzzle(num: int) -> Tuple[List[int], float, List[str]]:
    """生成谜题。

    参数:
        num: 数字个数 (>=2)

    返回:
        (nums, target, postfix_expression)
    """
    assert num >= 2, "数字个数必须 >= 2"

    # 如遇非法表达式（如除零），重新生成
    while True:
        nums = [random.randint(1, 9) for _ in range(num)]
        postfix_expression = _generate_postfix_expression(nums)
        try:
            target = _eval_postfix(postfix_expression)
            # 排除无穷大或非数字
            if target != float('inf') and target != float('-inf') and target == target:
                return nums, target, postfix_expression
        except ValueError:
            # 重新生成
            continue


def evaluate_response(response: str, nums: List[int], target: float, tol: float = 1e-6) -> bool:
    """解析 LLM 的回答并检查其是否正确。

    返回 True 表示表达式合法且答案正确，否则 False。
    """
    import re

    # 先尝试找 Answer: `...` 格式
    expr_match = re.search(r"Answer:\s*`([^`]+)`", response, re.IGNORECASE)
    if expr_match:
        expr = expr_match.group(1).strip()
    else:
        # 回退到 Answer: ... 无反引号
        expr_match2 = re.search(r"Answer:\s*(.*)", response, re.IGNORECASE)
        if expr_match2:
            expr = expr_match2.group(1).strip()
        else:
            expr = ""

    # print("[Debug] Extracted expression:", expr)

    if not expr:
        # print("[Debug] No expression found in response.")
        return False

    # 检查是否只用给定数字且各一次
    digits_in_expr = [int(ch) for ch in expr if ch.isdigit()]
    digit_usage_correct = (sorted(digits_in_expr) == sorted(nums))
    # print("[Debug] Digits used:", digits_in_expr)
    # print("[Debug] Expected digits:", nums)
    # print("[Debug] Digit usage correct:", digit_usage_correct)
    if not digit_usage_correct:
        return False

    # 安全计算表达式结果
    try:
        result = eval(expr)
        print("[Debug] Evaluated result:", result)
    except Exception as e:
        print("[Debug] Eval error:", e)
        return False

    diff = abs(result - target)
    print("[Debug] Target:", target, "| Diff:", diff)

    return diff < tol


def experiment(model_key: str, num: int):
    """运行一次实验，让 LLM 解谜。

    参数:
        model_key: 选定的 LLM 键。
        num: 数字个数。
    """
    # 先在此函数内部生成谜题
    nums, target, _ = generate_puzzle(num)
    nums_str = ', '.join(map(str, nums))
    # 任务描述
    task_desc = (
        "You are a math genius. Your task is to use each given number exactly once,"
        " combining them with +, -, *, / and parentheses to reach the target number."
        " Explain your reasoning step by step. When you finish, output the final expression"
        " in the exact format: Answer: `EXPR` (use backticks). Do not apply any other formatting inside the backticks.\n\n"
    )

    # One-shot example
    example_section = (
        "Here is an example:\n"
        "Numbers: 6, 3, 4\n"
        "Target: 14\n"
        "Reasoning:\n"
        "1. Multiply 6 and 3 to get 18.\n"
        "2. Subtract 4 to get 14.\n"
        "Answer: `(6*3)-4`\n\n"
    )

    # 真实问题
    problem_section = (
        "Here is the problem you need to solve:\n"
        f"Numbers: {nums_str}\n"
        f"Target: {target}\n"
        "Reasoning:"
    )

    prompt = task_desc + example_section + problem_section

    llm = ExampleLLM(model_key)
    response, prompt_tokens, completion_tokens = llm.call_llm(prompt)

    print("=== Puzzle ===")
    print(f"数字: {nums_str}")
    print(f"目标值: {target}")
    print("=== LLM Response ===")
    print(response.strip())
    print("=== Token Usage ===")
    print(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")

    is_correct = evaluate_response(response, nums, target)
    print("=== Evaluation ===")
    print("Correct" if is_correct else "Incorrect")

    return response.strip(), is_correct


# ---------------------- 批量实验与绘图 ----------------------


def run_experiments(model_key: str, num_start: int = 3, num_end: int = 8, trials: int = 50):
    """循环运行实验并绘制准确率折线图。

    Args:
        model_key: 示例 LLM 模型键。
        num_start: 起始数字个数（包含）。
        num_end: 结束数字个数（包含）。
        trials: 每个数字个数运行的实验次数。
    """
    import matplotlib.pyplot as plt

    nums_list = list(range(num_start, num_end + 1))
    accuracies = []

    for n in nums_list:
        success = 0
        for _ in tqdm(range(trials)):
            _, is_correct = experiment(model_key, n)
            if is_correct:
                success += 1
        acc = success / trials
        accuracies.append(acc)
        print(f"[Summary] n={n}: {success}/{trials} = {acc:.2%}")

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(nums_list, accuracies, marker='o')
    plt.xlabel('Number of digits (n)')
    plt.ylabel('Accuracy')
    plt.title(f'LLM accuracy over {trials} trials per n')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 批量运行并绘图
    run_experiments("o1-mini-1mtpm", 4, 8, 10) 
    # 3 0.98