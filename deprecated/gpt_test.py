import re
import random
import itertools

from llm_client import ExampleLLM

llm = ExampleLLM("o1-mini-1mtpm")

def extract_numbers(expr):
    # 提取等式中的所有整数
    return list(map(int, re.findall(r"\\d+", expr)))

def check_equation(expr, target):
    try:
        # 只允许加减乘除和括号
        allowed = set("0123456789+-*/() ")
        if not set(expr) <= allowed:
            return False
        return abs(eval(expr) - target) < 1e-6
    except Exception:
        return False

def has_solution(nums, target):
    """Universal solution function supporting arbitrary number of digits and target values"""
    ops = ['+', '-', '*', '/']
    n = len(nums)
    
    if n == 2:
        a, b = nums
        exprs = [f"{a}{op}{b}" for op in ops] + [f"{b}{op}{a}" for op in ops]
    elif n == 3:
        for ns in set(itertools.permutations(nums)):
            a, b, c = ns
            exprs = [
                f"({a}{op1}{b}){op2}{c}" for op1 in ops for op2 in ops
            ] + [
                f"{a}{op1}({b}{op2}{c})" for op1 in ops for op2 in ops
            ]
            for expr in exprs:
                try:
                    if abs(eval(expr) - target) < 1e-6:
                        return expr
                except Exception:
                    continue
        return None
    elif n == 4:
        for ns in set(itertools.permutations(nums)):
            a, b, c, d = ns
            exprs = [
                f"(({a}{op1}{b}){op2}{c}){op3}{d}" for op1 in ops for op2 in ops for op3 in ops
            ] + [
                f"({a}{op1}({b}{op2}{c})){op3}{d}" for op1 in ops for op2 in ops for op3 in ops
            ] + [
                f"({a}{op1}{b}){op2}({c}{op3}{d})" for op1 in ops for op2 in ops for op3 in ops
            ] + [
                f"{a}{op1}(({b}{op2}{c}){op3}{d})" for op1 in ops for op2 in ops for op3 in ops
            ] + [
                f"{a}{op1}({b}{op2}({c}{op3}{d}))" for op1 in ops for op2 in ops for op3 in ops
            ]
            for expr in exprs:
                try:
                    if abs(eval(expr) - target) < 1e-6:
                        return expr
                except Exception:
                    continue
        return None
    elif n == 5:
        for ns in set(itertools.permutations(nums)):
            a, b, c, d, e = ns
            exprs = [
                f"((({a}{op1}{b}){op2}{c}){op3}{d}){op4}{e}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"(({a}{op1}({b}{op2}{c})){op3}{d}){op4}{e}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"(({a}{op1}{b}){op2}({c}{op3}{d})){op4}{e}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"({a}{op1}(({b}{op2}{c}){op3}{d})){op4}{e}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"({a}{op1}({b}{op2}({c}{op3}{d}))){op4}{e}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"({a}{op1}{b}){op2}(({c}{op3}{d}){op4}{e})" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"({a}{op1}{b}){op2}({c}{op3}({d}{op4}{e}))" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"{a}{op1}((({b}{op2}{c}){op3}{d}){op4}{e})" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"{a}{op1}(({b}{op2}({c}{op3}{d})){op4}{e})" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"{a}{op1}({b}{op2}(({c}{op3}{d}){op4}{e}))" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ] + [
                f"{a}{op1}({b}{op2}({c}{op3}({d}{op4}{e})))" for op1 in ops for op2 in ops for op3 in ops for op4 in ops
            ]
            for expr in exprs:
                try:
                    if abs(eval(expr) - target) < 1e-6:
                        return expr
                except Exception:
                    continue
        return None
    elif n == 6:
        for ns in set(itertools.permutations(nums)):
            a, b, c, d, e, f = ns
            # Simplified expression patterns for 6 digits
            exprs = [
                f"((({a}{op1}{b}){op2}{c}){op3}{d}){op4}{e}){op5}{f}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops for op5 in ops
            ] + [
                f"({a}{op1}{b}){op2}({c}{op3}{d}){op4}({e}{op5}{f})" for op1 in ops for op2 in ops for op3 in ops for op4 in ops for op5 in ops
            ]
            for expr in exprs:
                try:
                    if abs(eval(expr) - target) < 1e-6:
                        return expr
                except Exception:
                    continue
        return None
    elif n == 7:
        for ns in set(itertools.permutations(nums)):
            a, b, c, d, e, f, g = ns
            # Simplified expression patterns for 7 digits
            exprs = [
                f"((({a}{op1}{b}){op2}{c}){op3}{d}){op4}{e}){op5}{f}){op6}{g}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops for op5 in ops for op6 in ops
            ]
            for expr in exprs:
                try:
                    if abs(eval(expr) - target) < 1e-6:
                        return expr
                except Exception:
                    continue
        return None
    elif n == 8:
        for ns in set(itertools.permutations(nums)):
            a, b, c, d, e, f, g, h = ns
            # Simplified expression patterns for 8 digits
            exprs = [
                f"((({a}{op1}{b}){op2}{c}){op3}{d}){op4}{e}){op5}{f}){op6}{g}){op7}{h}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops for op5 in ops for op6 in ops for op7 in ops
            ]
            for expr in exprs:
                try:
                    if abs(eval(expr) - target) < 1e-6:
                        return expr
                except Exception:
                    continue
        return None
    elif n == 9:
        for ns in set(itertools.permutations(nums)):
            a, b, c, d, e, f, g, h, i = ns
            # Simplified expression patterns for 9 digits
            exprs = [
                f"((({a}{op1}{b}){op2}{c}){op3}{d}){op4}{e}){op5}{f}){op6}{g}){op7}{h}){op8}{i}" for op1 in ops for op2 in ops for op3 in ops for op4 in ops for op5 in ops for op6 in ops for op7 in ops for op8 in ops
            ]
            for expr in exprs:
                try:
                    if abs(eval(expr) - target) < 1e-6:
                        return expr
                except Exception:
                    continue
        return None
    
    # For 2-digit case
    for expr in exprs:
        try:
            if abs(eval(expr) - target) < 1e-6:
                return expr
        except Exception:
            continue
    return None

def main():
    points = [18, 24, 30, 36, 42, 48, 54]
    count = 0
    tried = set()
    
    for point in points:
        num_count = point // 6
        print(f"\n--- {point} points ({num_count} digits) ---")
        point_count = 0
        point_total = 0
        point_correct = 0
        while point_count < 10:
            nums = tuple(sorted(random.sample(range(1, 14), num_count)))
            if nums in tried:
                continue
            tried.add(nums)
            expr = has_solution(nums, point)
            if not expr:
                continue
            nums_str = ','.join(map(str, nums))
            print(f"Group {count+1}: Numbers: {nums_str}")
            print(f"  Brute force solution: {expr}")
            solve_prompt = f"Please use addition, subtraction, multiplication, and division with parentheses to combine {nums_str} these {num_count} numbers into {point}, each number can only be used once, no repetition allowed, output the expression itself, no text, no bold expressions, no explanations, no equals sign, and only output oneline."
            for i in range(10):
                solve_expr, _, _ = llm.call_llm(solve_prompt)
                solve_expr = solve_expr.strip()
                print(f"  GPT model answer #{i+1}: {solve_expr}")
                is_valid = check_equation(solve_expr, point)
                point_total += 1
                if is_valid:
                    point_correct += 1
            print()
            count += 1
            point_count += 1
        
        print(f"{point} points accuracy: {point_correct}/{point_total} = {point_correct/point_total:.2%}")

if __name__ == "__main__":
    main() 
    