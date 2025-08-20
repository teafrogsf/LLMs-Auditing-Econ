from random import randint, shuffle, random
import networkx as nx

class Generator:
    def __init__(self, num_of_nodes=10, edge_probability=0.35, max_capacity=10):
        """
        num_of_nodes: 节点数量(编号为 0..n-1)
        edge_probability: 额外随机边(非骨干路径)的出现概率
        max_capacity: 边容量上限(下限为 1)
        """
        self.num_of_nodes = num_of_nodes
        self.edge_probability = edge_probability
        self.max_capacity = max_capacity

    def _rand_capacity(self):
        return randint(1, self.max_capacity)

    def generate_graph(self):
        """
        生成一个有向图 G,确保存在至少一条 s->t 路径。
        返回: G,(s,t)
        """

        while True:
            idx = list(range(self.num_of_nodes))
            shuffle(idx)
            # 随机决定骨干路径长度
            path_len = randint(2, self.num_of_nodes)
            path_nodes = idx[:path_len]
            s = path_nodes[0]
            t = path_nodes[-1]
            # 有向图 + 骨干路径
            G = nx.DiGraph()
            G.add_nodes_from(range(self.num_of_nodes))
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                G.add_edge(u, v, capacity=self._rand_capacity())
            # 随机添加额外有向边
            nodes = list(G.nodes())
            for u in nodes:
                for v in nodes:
                    if u == v:
                        continue
                    # 避免重复边：DiGraph 允许更新属性，这里只在不存在时添加
                    if not G.has_edge(u, v) and random() < self.edge_probability:
                        G.add_edge(u, v, capacity=self._rand_capacity())

            if nx.has_path(G, s, t):
                return G, (s, t)

    def generate(self):
        G, q = self.generate_graph()
        return G, (q[0], q[1])


# 使用示例：
# gen = Generator(num_of_nodes=12, edge_probability=0.3, max_capacity=8)
# G, (s, t) = gen.generate()
# print(s, t, G.number_of_nodes(), G.number_of_edges())
# # 边的容量：G[u][v]["capacity"]
