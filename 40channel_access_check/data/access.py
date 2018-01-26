"""
网络可达性测试的数据生成文件。主要生成判断网络可达性的相关数据集。
"""
import networkx as nx
from graphviz import Graph
import networkx.classes.graph as nxGraph
import random
import time
import os
import numpy as np


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


def rand_edges(graph: nxGraph, rm_ratio: float = 0.2) -> tuple:
    """
    从双向图graph中以rm_ratio的概率选出链路
    """
    if len(graph.edges()) is 0:
        print(graph.nodes())
        print(graph.edges())
        raise ValueError("empty graph!")
    else:
        rm_edges = []
        for edge in graph.edges():
            if random.random() < rm_ratio:
                rm_edges.append(edge)
        return tuple(rm_edges)


def get_label(src: int, dst: int, graph: nxGraph):
    """
    label为1表示在graph中，从src到dst存在路径，为0表示不存在路径
    """
    label = 1
    try:
        nx.algorithms.shortest_path(graph, src, dst)
    except nx.NetworkXNoPath:
        label = 0
    return label


def gen_img(graph: nxGraph,
            node_num,
            min_rm_ratio: float = 0.2,
            max_rm_ratio: float = 0.5,
            root_folder='train',
            source_folder='',
            source_file='train.txt',
            channel_num=40):
    """
    在graph上，以[min_rm_ratio, max_rm_ratio)区间的概率随机删除一些链路，然后用随机选取的一对节点作为源宿点，判断其是否连通。
    并将生成的图像放到root_folder目录中，对应的文件名称和标签放到source_folder目录下的train.txt文件中。

    :param graph: 输入的图
    :param node_num: 图的节点数
    :param root_folder: 根目录，存放图片文件
    :param source_folder: 存放图片名和label的目录
    :param source_file: 存放文件名和label的文件名
    :param channel_num: 表示图片的通道数
    :param min_rm_ratio: 删除链路的最小概率
    :param max_rm_ratio: 删除链路的最大概率
    """
    # 首先把文件前缀确定下来。
    name = str(time.time())
    total_label = 0
    src, dst = gen_src_dst(0, node_num)
    for channel in range(channel_num):
        rm_ratio = random.uniform(min_rm_ratio, max_rm_ratio)  # 随机从[min,max]中生成一个随机数
        # 生成原始数据
        rds = rand_edges(graph, rm_ratio)
        graph.remove_edges_from(rds)
        total_label += get_label(src, dst, graph)
        # 构造graphviz对象
        g = Graph(format='jpeg', engine='neato')
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
        g.render(filename=name+'-'+str(channel), directory=root_folder, view=False, cleanup=True)
        graph.add_edges_from(rds)  # 恢复刚才删掉的边。

    # 存储到文件中去
    file = open(os.path.join(source_folder, source_file), 'a')
    file.write(name + '.jpeg ' + str(total_label) + '\n')
    file.flush()
    file.close()


def data_gen_val():
    """
    验证数据生成的样例
    :return:
    """
    for node_num in range(10, 30):
        for index in range(100):
            for min_ratio in np.arange(0, 0.8, 0.05):
                max_ratio = min_ratio+0.2
                p = 2.5 / node_num
                net = gen_net_er(node_num, p)
                gen_img(net, node_num, min_rm_ratio=min_ratio, max_rm_ratio=max_ratio,
                        root_folder='/media/yby/knowledge/val', source_file='/media/yby/knowledge/val.txt',
                        channel_num=16)


def data_gen_train():
    """
    训练数据生成的样例
    :return:
    """
    for node_num in range(10, 30):
        for index in range(100):
            for min_ratio in np.arange(0, 0.8, 0.01):
                max_ratio = min_ratio+0.2
                p = 2.5 / node_num
                net = gen_net_er(node_num, p)
                gen_img(net, node_num, min_rm_ratio=min_ratio, max_rm_ratio=max_ratio,
                        root_folder='/media/yby/knowledge/train', source_file='/media/yby/knowledge/train.txt',
                        channel_num=16)
