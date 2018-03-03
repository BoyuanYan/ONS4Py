import networkx as nx
import os
import numpy as np
import graphviz as gz
from PIL import Image
import subprocess as sp
import io


file_prefix = "../resources"


class RwaNetwork(nx.Graph):
    """
    RWA network
    """
    def __init__(self, filename: str, wave_num: int):
        """

        :param filename: 标明网络的md文件，其中前两行是表头和md表格的标志“|:---|”，内容为index，src，dst，weight
        :param wave_num: 每条链路包含的波长个数
        """
        super(RwaNetwork, self).__init__()
        self.net_name = filename.split('.')[0]
        self.wave_num = wave_num
        filepath = os.path.join(file_prefix, filename)
        if os.path.isfile(filepath):
            datas = np.loadtxt(filepath, delimiter='|', skiprows=2, dtype=str)
            self.origin_data = datas[:, 1:(datas.shape[1]-1)]
            for i in range(self.origin_data.shape[0]):
                wave_avai = [True for i in range(wave_num)]
                self.add_edge(self.origin_data[i, 1], self.origin_data[i, 2],
                              weight=float(self.origin_data[i, 3]),
                              is_wave_avai=wave_avai)
        else:
            raise FileExistsError("file {} doesn't exists.".format(filepath))

    def gen_img(self, width: int, height: int, src: str, dst: str, mode: str) -> np.ndarray:
        """
        将网络当前的状态先生成channels张灰度图片，然后以CHW的格式表示出来
        :param width 生成图片后，resize图片到指定宽度
        :param height 生成图片后，resize图片到指定高度
        :param src 网络中到达业务的源点，为None表示不取源点，此时dst也应该为None
        :param dst 网络中到达业务的宿点，为None表示不取宿点，此时src也应该为None
        :param mode 返回状态的模式选择，如果为learning，则返回CHW的stacked灰度图像；如果为alg，则返回源宿点请求
        """
        if mode.startswith('alg'):
            return np.array([src, dst])
        elif mode.startswith('learning'):
            rtn = None
            for wave_index in range(self.wave_num):
                gz_graph = gz.Graph(format='png', engine='neato')
                gz_graph.attr('node', shape='point', fixedsize='true', height='0.1', width='0.1', label='')
                gz_graph.attr('edge')
                for node in self.nodes():
                    gz_graph.node(name=node)
                if src and dst:  # 如果src和dst都不是None
                    gz_graph.node(name=src, shape='triangle', height='0.2', width='0.2')
                    gz_graph.node(name=dst, shape='triangle', height='0.2', width='0.2')
                for edge in self.edges():
                    if self.get_edge_data(edge[0], edge[1])['is_wave_avai'][wave_index]:
                        gz_graph.edge(edge[0], edge[1])
                    else:
                        gz_graph.edge(edge[0], edge[1], color='white')
                img = Image.open(io.BytesIO(gz_graph.pipe()))  # 将gz_graph转化成RGB图像
                img = img.convert('L')  # 转灰度
                img = img.resize(size=(width, height))  # resize
                img = np.array(img)  # convert to np.array
                img = img / 255.0  # 归一化到[0-1]
                img = img[np.newaxis, :]  # add 1 dimension for channel
                if rtn is not None:
                    rtn = np.concatenate((rtn, np.array(img)), axis=0)
                else:
                    rtn = np.array(img)

            return rtn
        else:
            raise ValueError("wrong mode parameter")

    def set_wave_state(self, wave_index, nodes: list, state: bool, check: bool=True):
        """
        设置一条路径上的某个波长的可用状态
        :param wave_index: 编号从0开始
        :param nodes: 路径经过的节点序列
        :param state: 要设置的状态
        :param check: 是否检验状态
        :return:
        """
        assert len(nodes) >= 2
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            if check:
                assert self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] != state
            self.get_edge_data(start_node, end_node)['is_wave_avai'][wave_index] = state
            start_node = end_node

    def get_avai_waves(self, nodes: list) -> list:
        """
        获取指定路径上的可用波长下标
        :param nodes: 路径经过的节点序列
        :return:
        """
        rtn = np.array([True for i in range(self.wave_num)])
        assert len(nodes) >= 2
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            rtn = np.logical_and(rtn,
                                 np.array(self.get_edge_data(start_node, end_node)['is_wave_avai']))
            start_node = end_node
        return np.where(rtn == True)[0].tolist()

    def exist_rw_allocation(self, path_list: list) -> [bool, int, int]:
        """
        扫描path_list中所有路径上的所有波长，按照FirstFit判断是否存在可分配方案
        :param path_list:
        :return: 是否存在路径，路径index，波长index
        """
        if len(path_list) == 0 or path_list[0] is None:
            return False, -1, -1

        for path_index, nodes in enumerate(path_list):
            edges = self.extract_path(nodes)
            # print(edges)
            for wave_index in range(self.wave_num):
                is_avai = True
                for edge in edges:
                    if self.get_edge_data(edge[0], edge[1])['is_wave_avai'][wave_index] is False:
                        is_avai = False
                        break
                if is_avai is True:
                    return True, path_index, wave_index

        return False, -1, -1

    def is_allocable(self, path: list, wave_index: int) -> bool:
        """
        判断路由path上wave_index波长的路径是否可分配。
        :param path:
        :param wave_index:
        :return:
        """
        edges = self.extract_path(path)
        is_avai = True
        for edge in edges:
            if self.get_edge_data(edge[0], edge[1])['is_wave_avai'][wave_index] is False:
                is_avai = False
                break
        return is_avai

    def extract_path(self, nodes):
        assert len(nodes) >= 2
        rtn = []
        start_node = nodes[0]
        for i in range(1, len(nodes)):
            end_node = nodes[i]
            rtn.append((start_node, end_node))
            start_node = end_node
        return rtn

