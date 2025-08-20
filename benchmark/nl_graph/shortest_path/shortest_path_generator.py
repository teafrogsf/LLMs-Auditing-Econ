from random import randint, shuffle, random
import networkx as nx

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

