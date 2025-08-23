import os
import networkx as nx
import re
from loguru import logger

def translate(G, q, pattern:str):
    edge = list(G.edges())
    n, m = G.number_of_nodes(), G.number_of_edges()
    Q = ''
    if pattern in ["cot", "k-shot"]:
        prompt_file_path = os.path.join(os.path.dirname(__file__), pattern + "_prompt.txt")
        with open(prompt_file_path, "r") as f:
            exemplar = f.read()
        Q = Q + exemplar + "\n\n\n"
    Q = Q + "In a directed graph, the nodes are numbered from 0 to " + str(n-1)+", and the edges are:\n"
    for i in range(len(edge)):
        Q = Q + 'an edge from node '+str(edge[i][0])+' to node '+str(edge[i][1]) + " with capacity " + str(G[edge[i][0]][edge[i][1]]["capacity"])
        if i + 1 == len(edge):
            Q = Q + '.'
        else:
            Q = Q + ','
        Q = Q + '\n'
    Q = Q + "Q: What is the maximum flow from node " + str(q[0])+" to node " + str(q[1]) + "?"
    Q = Q + "\nA:"
    return Q


def evaluate(ans, G, q, std):
    """
    从 <answer>...</answer> 中提取最终答案的数字并与标准答案 std 做比较。
    """
    # 提取标签中的数字
    pattern = re.compile(r"<\s*answer\s*>\s*([+-]?\d+(?:\.\d+)?)\s*<\s*/\s*answer\s*>",
                         flags=re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(ans)
    if not matches:
        # logger.debug("没有在 <answer> 标签中找到数值")
        return 0.0

    raw = matches[-1]  # 若有多个答案，取最后一个 
    # logger.debug(f"模型答案：{raw}")
    # 2) 将数值转为 float
    try:
        num = float(raw)
    except ValueError:
        # logger.debug(f"提取到的不是合法数字：{raw}")
        return 0.0

    if std == 0:
        return 1 if abs(num) == 0 else 0

    if num <= std:
        score = num / std
    else:
        # logger.debug("回答大于标准答案")
        score = 0.0

    return score