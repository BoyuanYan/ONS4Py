from RwaNet import k_shortest_paths
from Service import RwaGame
import argparse

parser = argparse.ArgumentParser(
    description='GRWA Training')

parser.add_argument('--net', type=str, default='6node.md',
                    help="网络拓扑图，默认在resources目录下搜索")
parser.add_argument('--wave-num', type=int, default=10,
                    help='拓扑中每条链路的波长数')
parser.add_argument('--rou', type=int, default=5,
                    help='业务到达的平均间隔，泊松分布')
parser.add_argument('--miu', type=int, default=100,
                    help='业务持续的平均时间，泊松分布')
parser.add_argument('--max-iter', type=int, default=10000,
                    help='一次episode中，分配的最大业务数量')
parser.add_argument('--k', type=int, default=1,
                    help='RWA算法中，采取ksp计算路由的k值')
parser.add_argument('--mode', type=str, default='alg',
                    help='RWA执行的模式，alg表示使用ksp+FirstFit，learning表示学习模式')
parser.add_argument('--img-width', type=int, default=224,
                    help="生成的网络灰度图的宽度")
parser.add_argument('--img-height', type=int, default=224,
                    help="生成的网络灰度图的高度")
parser.add_argument('--weight', type=str, default='None',
                    help='计算路由的时候，以什么属性为权重')

parser.add_argument('--episode', type=int, default=1e6,
                    help='')


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
        test_ksp(args, weight)
        return

    envs = RwaGame(net_config=args.net, wave_num=args.wave_num, rou=args.rou, miu=args.miu,
                   max_iter=args.max_iter, k=args.k, mode=args.mode, img_width=args.img_width,
                   img_height=args.img_height, weight=weight)
    observation, reward, done, _ = envs.reset()


def test_ksp(args, weight):
    """
    使用ksp+FirstFit算法测试
    :param args:
    :return:
    """
    succ_count = 0
    fail_count = 0

    envs = RwaGame(net_config=args.net, wave_num=args.wave_num, rou=args.rou, miu=args.miu,
                   max_iter=args.max_iter, k=args.k, mode=args.mode, img_width=args.img_width,
                   img_height=args.img_height, weight=weight)
    observation, reward, done, _ = envs.reset()
    while not done:
        if observation[0] is not None:
            # 如果当前时间有业务到达
            src, dst = observation
            path_list = k_shortest_paths(envs.net, src, dst, k=args.k, weight=weight)

            exist, path_index, wave_index = envs.net.exist_rw_allocation(path_list)
            if exist:
                # 如果有可用分配方案
                action = path_index*args.wave_num + wave_index
                succ_count += 1
            else:
                # 如果没有可用分配方案
                action = args.wave_num*args.k
                fail_count += 1
        else:
            # 如果当前时间没有业务到达
            action = args.wave_num*args.k
        observation, reward, done, _ = envs.step(action)

    total = succ_count+fail_count
    print("一共{}条业务，其中分配成功{}条，分配失败{}条，阻塞率{:.4f}".format(total, succ_count, fail_count, fail_count/total))


if __name__ == "__main__":
    main()