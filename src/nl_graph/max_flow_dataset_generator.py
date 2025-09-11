import json
import sys
import os
import random
from loguru import logger

from max_flow_generator import Generator
import networkx as nx

logger.add("logs/max_flow.log", rotation="10 MB", retention="7 days", level="INFO")

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def generate_unique_graphs(num_graphs=1000, seed=None):
    """
    生成指定数量的不重复图
    
    Args:
        num_graphs: 要生成的图数量
        seed: 随机种子，如果为None则使用全局设置的种子
    """
    if seed is not None:
        random.seed(seed)
    
    n_min = 15
    n_max = 20
    num_nodes = random.randint(n_min, n_max)  # 随机节点数
    prob_min = 0.25
    prob_max = 0.3  # 边概率
    edge_prob = random.uniform(prob_min, prob_max)
    generator = Generator(num_of_nodes=num_nodes, edge_probability=edge_prob, max_capacity=20)
    graphs_data = {}
    generated_graphs = set()  # 用于存储已生成图的字符串表示，确保不重复
    
    count = 0
    attempts = 0
    max_attempts = num_graphs * 3  # 最大尝试次数
    
    logger.info(f"开始生成 {num_graphs} 个不重复的图...")
    
    while count < num_graphs and attempts < max_attempts:
        attempts += 1
        
        # 生成图
        G, (s, t) = generator.generate()
        edges_with_capacity = []
        for u, v, data in G.edges(data=True):
            edges_with_capacity.append((u, v, data['capacity']))
        edges_with_capacity.sort()  # 排序确保一致性
        
        # 将图转换为字符串
        graph_signature = str((G.number_of_nodes(), tuple(edges_with_capacity), s, t))
        
        # 检查是否重复
        if graph_signature not in generated_graphs:
            generated_graphs.add(graph_signature)
            
            # 使用nx.node_link_data转换图数据
            graph_data = nx.node_link_data(G)
            graph_data_dump = json.dumps(graph_data, indent=None, separators=(',', ':'))
            
            graphs_data[count] = {
                "graph": graph_data_dump,
                "source": s,
                "target": t
            }
            count += 1
            
            if count % 100 == 0:
                logger.info(f"已生成 {count} 个图...")
    
    if count < num_graphs:
        logger.warning(f"警告：只生成了 {count} 个不重复的图（目标：{num_graphs}）")
    else:
        logger.info(f"成功生成 {count} 个不重复的图")
    
    return graphs_data

def save_graphs_to_json(graphs_data, filename='max_flow_graphs.json'):
    """
    将图数据保存到JSON文件
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(graphs_data, f, indent=1)
    logger.info(f"图数据已保存")

if __name__ == "__main__":
    # 生成1000个不重复的图
    graphs_data = generate_unique_graphs(1000, seed=RANDOM_SEED)
    
    # 保存到JSON文件
    save_graphs_to_json(graphs_data)
    
    logger.info(f"总共生成了 {len(graphs_data)} 个图")