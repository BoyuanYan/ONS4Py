"""
RWA光网络的数据生成文件。主要生成给出路由选择的相关数据集。
数据以np.ndarray的形式存储，格式是CHW，其中C=40
"""
import networkx as nx
from graphviz import Graph
import random
import time
import os
import numpy as np
import cv2


def gen_net_ba(node_num, m):
    """
    复杂网络：BA无标度网络
    node_num: Number of nodes
    m: Number of edges to attach from a new node to existing nodes
    """
    return nx.generators.barabasi_albert_graph(node_num, m)


def gen_net_ws(node_num, m, p=0.3):
    """
    复杂网络：WS小世界网络
    node_num: Number of nodes
    m: Number of edges to attach from a new node to existing nodes
    p: random reconnection probability
    """
    return nx.generators.watts_strogatz_graph(node_num, m, p)


def gen_net_er(node_num, p):
    """
    规则网络：ER随机图。有可能生成不连通图
    :param node_num:
    :param p:
    :return:
    """
    net = nx.generators.erdos_renyi_graph(node_num, p)
    while(True):
        subs = nx.connected_components(net)
        if sum(1 for _ in subs) is 1:  # 如果只有一个连通子图，则说明本图是连通图
            break
        net = nx.generators.erdos_renyi_graph(node_num, p)
    return net


def gen_src_dst(node_begin, node_end):
    """
    在[node_begin, node_end)中随机选择两个节点作为源点和宿点
    """
    src = random.randint(node_begin, node_end - 1)
    dst = random.randint(node_begin, node_end - 1)
    while src == dst:
        dst = random.randint(node_begin, node_end - 1)
    return src, dst


def rand_edges(graph: Graph, rm_ratio: float = 0.2) -> tuple:
    """
    从双向图graph中以rm_ratio的概率选出链路
    """
    if len(graph.edges()) is 0:
        raise ValueError("empty graph!")
    else:
        rm_edges = []
        for edge in graph.edges():
            if random.random() < rm_ratio:
                rm_edges.append(edge)
        return tuple(rm_edges)


def get_label(src: int, dst: int, graph: Graph):
    """
    label为1表示在graph中，从src到dst存在路径，为0表示不存在路径
    """
    label = 1
    try:
        nx.algorithms.shortest_path(graph, src, dst)
    except nx.NetworkXNoPath:
        label = 0
    return label


def gen_img(graph: Graph, node_num, rm_ratio: float = 0.2,
            root_folder='train',
            source_folder='',
            source_file='train.txt'):
    """
    在graph上，以rm_ratio的概率随机删除一些链路，然后用随机选取的一对节点作为源宿点，判断其是否联通。并将生成的图像放到root_folder目录中，
    对应的文件名称和标签放到source_folder目录下的train.txt文件中。
    """
    # 生成原始数据
    rds = rand_edges(graph, rm_ratio)
    graph.remove_edges_from(rds)
    src, dst = gen_src_dst(0, node_num)
    label = get_label(src, dst, graph)
    # 构造graphviz对象
    name = str(time.time())
    g = Graph(format='png', engine='neato')
    g.attr('node', shape='circle')
    g.attr('edge')
    for node in graph.nodes():
        g.node(name=str(node))
    g.node(name=str(src), shape='triangle')
    g.node(name=str(dst), shape='triangle')
    for edge in graph.edges():
        g.edge(str(edge[0]), str(edge[1]))
    for edge in rds:
        g.edge(str(edge[0]), str(edge[1]), color='white')
    g.render(filename=name, directory=root_folder, view=False, cleanup=True)
    # 存储到文件中去
    file = open(os.path.join(source_folder, source_file), 'a')
    file.write(name + '.png ' + str(label) + '\n')
    file.flush()
    file.close()


def data_gen_val():
    """
    验证数据生成的样例
    :return:
    """
    for node_num in range(10, 30):
        for index in range(1):
            for ratio in np.arange(0, 1, 0.5):
                p = 2.5 / node_num
                net = gen_net_er(node_num, p)
                gen_img(net, node_num, rm_ratio=ratio, root_folder='val', source_file='val.txt')


def data_gen_train():
    """
    训练数据生成的样例
    :return:
    """
    for node_num in range(10, 30):
        for index in range(1):
            for ratio in np.arange(0, 1, 0.01):
                p = 2.5 / node_num
                net = gen_net_er(node_num, p)
                gen_img(net, node_num, rm_ratio=ratio, root_folder='train', source_file='train.txt')