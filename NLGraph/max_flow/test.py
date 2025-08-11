import os
import sys


# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from flow import translate, evaluate


G = 0
q = (4,3)
ans = "The maximum flow from node 4 to node 3 is **6** units."
correct_answer = "6"
score = evaluate(ans,G,q, correct_answer)

