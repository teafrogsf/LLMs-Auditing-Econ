import random
import itertools
import operator

# 支持的运算符
ops = [('+', operator.add), ('-', operator.sub), ('*', operator.mul), ('/', operator.truediv)]

# 生成所有可能的表达式

def valid_24(nums):
    """
    判断给定4个数能否通过加减乘除得到24
    """
    def helper(nums):
        if len(nums) == 1:
            return abs(nums[0] - 24) < 1e-6
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
                    for sym, op in ops:
                        # 避免除以0
                        if sym == '/' and abs(nums[j]) < 1e-6:
                            continue
                        try:
                            val = op(nums[i], nums[j])
                        except:
                            continue
                        if helper(rest + [val]):
                            return True
        return False
    return helper(nums)


def generate_hard_24_problem():
    """
    随机生成一个较难的24点题目（低级模型难以解决）
    """
    # 先生成所有能算出24的组合
    candidates = []
    for nums in itertools.product(range(1, 11), repeat=4):
        if valid_24(list(nums)):
            candidates.append(nums)
    # 随机选一个
    while True:
        nums = random.choice(candidates)
        # 简单难度过滤：不能有3个及以上相同数字，不能有1、2、10等容易数字
        if len(set(nums)) >= 3 and all(x not in [1, 2, 10] for x in nums):
            return nums


def check_24_answer(nums, expr):
    """
    检查表达式expr是否用nums中的每个数各一次且结果为24
    """
    try:
        # 检查是否只用这四个数
        used = [int(s) for s in expr if s.isdigit()]
        if sorted(used) != sorted(nums):
            return False
        val = eval(expr)
        return abs(val - 24) < 1e-6
    except:
        return False 