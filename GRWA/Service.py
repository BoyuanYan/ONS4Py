from .RwaNet import RwaNetwork, k_shortest_paths
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
import numpy as np
import queue


modes = ['alg', 'learning']


class RwaGame(object):
    """
    RWA game, 模仿gym的实现
    """

    def __init__(self, net_config: str, wave_num: int, rou: float, miu: float,
                 max_iter: int, k: int, mode: str, img_width: int, img_height: int,
                 weight):
        """

        :param net_config: 网络配置文件
        :param wave_num: 链路波长数，CWDM是40， DWDM是80
        :param rou: 平均隔多少时间单位到达一条业务
        :param miu: 一条业务平均会在网络中存在多少时间单位后会离去
        :param max_iter: 一次episode中最大的轮数，即一次仿真的最大业务数
        :param mode: 模式，分为alg和learning两种，前者表示使用ksp+firstfit分配，后者表示使用rl算法学习
        :param img_width: 游戏界面的宽度
        :param img_height: 游戏界面的高度
        """
        super(RwaGame, self).__init__()
        self.net_config = net_config
        self.wave_num = wave_num
        self.img_width = img_width
        self.img_height = img_height
        self.weight = weight
        self.rou = rou
        self.miu = miu
        self.erl = miu / rou
        self.max_iter = max_iter
        self.k = k
        self.action_space = Discrete(k*wave_num+1)  # 最后一个值表示主动阻塞
        if mode in modes:
            self.mode = mode
        else:
            raise ValueError("wrong mode parameter.")
        # 一旦游戏开始，iter和time都指向当前的event下标和时间点。
        self.event_iter = 0
        self.time = 0
        self.net = RwaNetwork(self.net_config, wave_num=self.wave_num)
        self.services = {}
        self.events = []  # time_point, service_index, is_arrival_event

    def gen_src_dst(self):
        nodes = list(self.net.nodes())
        assert len(nodes) > 1
        src_index = np.random.randint(0, len(nodes))
        dst_index = np.random.randint(0, len(nodes))
        while src_index == dst_index:
            dst_index = np.random.randint(0, len(nodes))
        return nodes[src_index], nodes[dst_index]

    def reset(self):
        """
        reset environment
        :return:
        """
        self.event_iter = 0
        self.time = 0
        self.services = {}
        self.events = []
        self.net = RwaNetwork(self.net_config, wave_num=self.wave_num)

        base_time = 0
        for base_index in range(self.max_iter):
            src, dst = self.gen_src_dst()
            arrival = np.random.poisson(lam=self.rou) + base_time
            leave = np.random.poisson(lam=self.miu) + arrival
            self.services[base_index] = Service(base_index, src, dst, arrival, leave)
            self.events.append([arrival, base_index, True])
            self.events.append([leave, base_index, False])

            base_time = arrival
        self.events.sort(key=lambda time: time[0])

        # 返回第一个业务请求的状态
        src, dst = self.services[0].src, self.services[0].dst
        observation = self.net.gen_img(self.img_width, self.img_height, src, dst)
        reward = 1
        done = False
        info = None
        self.time = self.services[0].arrival_time
        return observation, reward, done, info



    def render(self):
        """
        渲染当前环境，返回当前环境的图像
        :return:
        """
        print("doesn't support")

    def step(self, action) -> [object, float, bool, dict]:
        """
        在当前时间点self.time,执行行为action，获取reward，并且转向下一状态。
        :param action:
        :return:
        """
        done = False
        # 首先，判断当前的处境，该时间点是否有业务到达或者离去，如果有，有几个
        if self.events[self.event_iter][0] > self.time:
            # 如果该时间点没有到达或者离去的业务，则action选什么都无所谓
            if action == self.k * self.wave_num:
                # 如果主动阻塞
                reward = 1
            else:
                # 如果选择其他行为，虽然没用，但是还是要惩罚
                reward = 0
        elif self.events[self.event_iter][0] == self.time:
            # 如果该时间点恰巧有业务到达或者离去
            # TODO 处理当前时间点的业务，并且将self.event_iter指向下一个要处理的事件
        else:
            # 如果该时间点之前还有没处理完的业务
            raise EnvironmentError("时间推进过程中，有漏掉未处理的事件")

        # 其次，判断是否已经走到了头
        if self.event_iter == len(self.events):
            # 如果已经把事件全部处理完，
            done = True
            observation = self.net.gen_img(self.img_width, self.img_height, None, None)
            return observation, reward, done, None

        # 第三，开始进行下一状态的处理
        self.time += 1  # 时间推进一个单位
        if self.events[self.event_iter][0] > self.time:
            # 如果该时间点没有到达或者离去的业务，则返回正常拓扑图
            observation = self.net.gen_img(self.img_width, self.img_height, None, None)
        elif self.events[self.event_iter][0] == self.time:
            # 如果该时间点恰巧有业务到达或者离去
            # TODO 处理当前时间点排在到达业务之前的离去业务，并将self.event_iter指向下一个要处理的事件
        else:
            # 如果该时间点之前还有没处理完的业务
            raise EnvironmentError("时间推进过程中，还有漏掉未处理的事件")

        return observation, reward, done, None

class Service(object):
    def __init__(self, index: int, src: str, dst: str,
                 arrival_time: int, leave_time: int):
        super(Service, self).__init__()
        self.index = index
        self.src = src
        self.dst = dst
        self.arrival_time = arrival_time
        self.leave_time = leave_time

    def add_allocation(self, path:list, wave_index: int):
        self.path = path
        self.wave_index = wave_index


def cmp(x, y):
    if x[0] < y[0]:
        return -1
    if x[0] > y[0]:
        return 1
    return 0