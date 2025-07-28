import os
from random import randint, shuffle, random
import networkx as nx
import argparse
from tqdm import tqdm

class Generator:
    def __init__(self,num_of_nodes = 10, edge_probability = 0.35, max_weight = 4):
        self.num_of_nodes = num_of_nodes
        self.edge_probability = edge_probability
        self.max_weight = max_weight

    def generate_graph(self):
        l = randint(2, 6)
        while True:
            idx = list(range(self.num_of_nodes))
            shuffle(idx)
            G = nx.Graph()
            G.add_nodes_from(range(self.num_of_nodes))
            for u in list(G.nodes()):
                for v in list(G.nodes()):
                    if u < v and random() < self.edge_probability:
                        weight = randint(1,self.max_weight)
                        G.add_edge(idx[u], idx[v], weight = weight)
            if nx.is_connected(G):
                q = []
                shuffle(idx)
                for u in list(G.nodes()):
                    if len(q) > 0:
                        break
                    for v in list(G.nodes()):
                        if u != v and not G.has_edge(idx[u], idx[v]) and nx.shortest_path_length(G, source=idx[u], target=idx[v])>=l:
                            q = [idx[u],idx[v]]
                            break
                if len(q) > 0:
                    return G, q
    def generate(self):
        G, q = self.generate_graph()
        return G, q

if __name__ == "__main__":
    # 简化为只生成一张简单模式的图
    print("正在生成一张简单模式的图...")
    
    # 使用简单模式的固定参数
    n_min = 5  # 简单模式：节点数在5-10之间
    n_max = 10
    edge_probability = 0.6 # 简单模式的边概率
    max_weight = 4  # 简单模式的最大权重
    
    # 创建生成器并生成图
    generator = Generator(num_of_nodes=num_of_nodes, edge_probability=edge_probability, max_weight=max_weight)
    Graph, q = generator.generate()
    
    # 输出图的基本信息
    print(f"生成的图有 {Graph.number_of_nodes()} 个节点，{Graph.number_of_edges()} 条边")
    print(f"查询的起点和终点：{q[0]} -> {q[1]}")
    
    # 显示图的边和权重
    print("\n图的边和权重：")
    edge = list(Graph.edges())
    for i in range(len(Graph.edges())):
        u, v = edge[i]
        weight = Graph[u][v]["weight"]
        print(f"边 {u}-{v}: 权重 {weight}")
    
    print("\n图生成完成！")
