from Service import RwaGame
from model import MobileNetV2
from subproc_env import SubprocEnv
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description='GRWA Training')

parser.add_argument('--mode', type=str, default='alg',
                    help='RWA执行的模式，alg表示使用ksp+FirstFit，learning表示学习模式')
parser.add_argument('--workers', type=int, default=4,
                    help='默认同步执行多少个游戏。')
parser.add_argument('--steps', type=int, default=10000,
                    help="训练总共要进行的步骤数")
#  RWA相关参数
parser.add_argument('--net', type=str, default='6node.md',
                    help="网络拓扑图，默认在resources目录下搜索")
parser.add_argument('--wave-num', type=int, default=10,
                    help='拓扑中每条链路的波长数')
parser.add_argument('--rou', type=int, default=5,
                    help='业务到达的平均间隔，泊松分布')
parser.add_argument('--miu', type=int, default=100,
                    help='业务持续的平均时间，泊松分布')
parser.add_argument('--max-iter', type=int, default=1000,
                    help='一次episode中，分配的业务数量')
parser.add_argument('--k', type=int, default=1,
                    help='RWA算法中，采取ksp计算路由的k值')
parser.add_argument('--img-width', type=int, default=224,
                    help="生成的网络灰度图的宽度")
parser.add_argument('--img-height', type=int, default=224,
                    help="生成的网络灰度图的高度")
parser.add_argument('--weight', type=str, default='None',
                    help='计算路由的时候，以什么属性为权重')
# RL算法相关参数
parser.add_argument('--episode', type=int, default=1e6,
                    help='一共训练多少次游戏')
parser.add_argument('--arch', type=str, default='mobilenet_v2',
                    help='采用的神经网络，默认是mobilenet_v2，目前只支持这一个')
parser.add_argument('--base-lr', type=float, default=5e-4,
                    help='起始learning rate值')
parser.add_argument('--lr-adjust', type=str, default='constant',
                    help='learning rate的调整策略，包括constant，exp，linear')
parser.add_argument('--alpha', type=float, default=0.99,
                    help='')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='')

parser.add_argument('--gamma', type=float, default=0.99,
                    help='')
parser.add_argument('--use-gae', type=bool, default=False,
                    help='https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/issues/49')



def main():
    """
    主程序
    :return:
    """
    args = parser.parse_args()
    if args.weight.startswith('None'):
        weight = None
    else:
        weight = args.weight
    if args.mode.startswith('alg'):
        ksp(args, weight)
        return

    envs = RwaGame(net_config=args.net, wave_num=args.wave_num, rou=args.rou, miu=args.miu,
                   max_iter=args.max_iter, k=args.k, mode=args.mode, img_width=args.img_width,
                   img_height=args.img_height, weight=weight)
    observation, reward, done, _ = envs.reset()


def make_env(net_config: str, wave_num: int, rou: float, miu: float,
                 max_iter: int, k: int, mode: str, img_width: int, img_height: int,
                 weight):
    def _thunk():
        rwa_game = RwaGame(net_config=net_config, wave_num=wave_num, rou=rou, miu=miu,
                           max_iter=max_iter, k=k, mode=mode, img_width=img_width,
                           img_height=img_height, weight=weight)
        return rwa_game
    return _thunk


def ksp(args, weight):
    """
    使用ksp+FirstFit算法测试
    :param args:
    :param weight:
    :return:
    """

    succ_count = [0 for i in range(args.workers)]
    fail_count = [0 for i in range(args.workers)]
    rewards_count = [0 for i in range(args.workers)]
    step_count = 0

    envs = [make_env(net_config=args.net, wave_num=args.wave_num, rou=args.rou, miu=args.miu,
                     max_iter=args.max_iter*(i+1), k=args.k, mode=args.mode, img_width=args.img_width,
                     img_height=args.img_height, weight=weight) for i in range(args.workers)]

    envs = SubprocEnv(envs)

    observation, reward, done, _ = envs.reset()

    while True:
        actions = []
        # 如果没有全部结束
        path_list = envs.k_shortest_paths(observation)
        exist, path_index, wave_index = envs.exist_rw_allocation(path_list)
        for rank in range(args.workers):
            if bool(done[rank]) is True:
                # 如果该进程的游戏已经结束了
                actions.append(-1)
            else:
                if observation[rank][0] is not None:
                    # 如果当前时间有业务到达
                    if exist[rank]:
                        # 如果有可用分配方案
                        actions.append(path_index[rank]*args.wave_num + wave_index[rank])
                        succ_count[rank] += 1
                    else:
                        # 如果没有可用分配方案
                        actions.append(args.wave_num*args.k)
                        fail_count[rank] += 1
                else:
                    # 如果当前时间没有业务到达
                    actions.append(args.wave_num*args.k)
            rewards_count[rank] += reward[rank]  # 计算reward总和

        envs.step_async(actions)
        observation, reward, done, _ = envs.step_wait()
        step_count += 1
        if step_count == args.steps:
            break

    envs.close()

    for i in range(args.workers):
        total = succ_count[i] + fail_count[i]
        print("rank {}: 一共{}条业务，其中分配成功{}条，分配失败{}条，阻塞率{:.4f}".format(i, total, succ_count[i],
                                                                    fail_count[i], fail_count[i]/total))
        print("reward是：{}".format(rewards_count[i]))


if __name__ == "__main__":
    main()