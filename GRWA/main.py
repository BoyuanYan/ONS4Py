import os
from Service import RwaGame
from model import MobileNetV2
from subproc_env import SubprocEnv
from storage import RolloutStorage
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

parser = argparse.ArgumentParser(
    description='GRWA Training')

parser.add_argument('--mode', type=str, default='alg',
                    help='RWA执行的模式，alg表示使用ksp+FirstFit，learning表示CNN学习模式, fcl表示FC学习模式，lstml表示LSTM学习模式')
parser.add_argument('--workers', type=int, default=16,
                    help='默认同步执行多少个游戏，默认值16')
parser.add_argument('--steps', type=int, default=10000,
                    help="所有游戏进程的训练总共要进行的步骤数")
parser.add_argument('--save-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--save-interval', type=int, default=100,
                    help='save interval, one save per n updates (default: 100)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
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
parser.add_argument('--num-steps', type=int, default=5,
                    help='number of forward steps in A2C (default: 5)')
parser.add_argument('--base-lr', type=float, default=7e-4,
                    help='起始learning rate值')
parser.add_argument('--lr-adjust', type=str, default='constant',
                    help='learning rate的调整策略，包括constant，exp，linear')
parser.add_argument('--alpha', type=float, default=0.99,
                    help='RMSprop optimizer apha (default: 0.99)')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                    help='max norm of gradients (default: 0.5)')

parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--use-gae', type=bool, default=False,
                    help='https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/issues/49')

args = parser.parse_args()


def main():
    """
    主程序
    :return:
    """
    num_cls = args.wave_num * args.k + 1  # 所有的路由和波长选择组合，加上啥都不选
    num_updates = args.steps // args.workers // args.num_steps  # 梯度一共需要更新的次数
    # 解析weight
    if args.weight.startswith('None'):
        weight = None
    else:
        weight = args.weight
    # 创建actor_critic
    if args.mode.startswith('alg'):
        ksp(args, weight)
        return
    elif args.mode.startswith('learning'):
        # CNN学习模式下，osb的shape应该是CHW
        obs_shape = (args.wave_num, args.img_height, args.img_width)
        actor_critic = MobileNetV2(in_channels=args.wave_num, num_classes=num_cls, t=6)
        optimizer = optim.RMSprop(actor_critic.parameters(), lr=args.base_lr, eps=args.epsilon, alpha=args.alpha)
    else:
        raise NotImplementedError

    # 创建游戏环境
    envs = [make_env(net_config=args.net, wave_num=args.wave_num, rou=args.rou, miu=args.miu,
                     max_iter=args.max_iter, k=args.k, mode=args.mode, img_width=args.img_width,
                     img_height=args.img_height, weight=weight) for i in range(args.workers)]
    envs = SubprocEnv(envs)
    # 创建游戏运行过程中相关变量存储更新的容器
    rollout = RolloutStorage(num_steps=args.num_steps, num_processes=args.workers,
                             obs_shape=obs_shape, action_shape=num_cls)
    current_obs = torch.zeros(args.workers, *obs_shape)

    observation, _, _, _ = envs.reset()
    update_current_obs(current_obs, observation)

    rollout.observations[0].copy_(current_obs)
    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.workers, 1])
    final_rewards = torch.zeros([args.workers, 1])

    start = time.time()

    for updata_i in range(num_updates):
        for step in range(args.num_steps):
            # 选择行为
            inp = Variable(rollout.observations[step], volatile=True)  # 禁止梯度更新
            value, action, action_log_prob = actor_critic.act(inputs=inp, deterministic=False)
            # 压缩维度，放到cpu上执行。因为没有用到GPU，所以并没有什么卵用，权当提示
            cpu_actions = action.data.squeeze(1).cpu().numpy()
            # actor_critic.act 得到的是action变量，需要将其转换成one_hot形式
            one_hot_action = torch.zeros(action.size()[0], num_cls).scatter_(1, action.data, 1)
            # 观察observation，以及下一个observation
            envs.step_async(cpu_actions)
            obs, reward, done, info = envs.step_wait()  # reward和done都是(n,)向量
            reward = torch.from_numpy(np.expand_dims(reward, 1)).float()
            episode_rewards += reward  # 累加reward分数
            # 如果游戏结束，则重新开始计算episode_rewards和final_rewards，并且以返回的reward为初始值重新进行累加。
            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])  # True --> 0, False --> 1
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks
            # 给masks扩充2个维度，与current_obs相乘。则运行结束的游戏进程对应的obs值会变成0，图像上表示全黑，即游戏结束的画面。
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
            update_current_obs(current_obs=current_obs, obs=obs)
            # print("final_rewards is {}".format(final_rewards))
            # print("mask is {}".format(masks))
            # 把本步骤得到的结果存储起来
            rollout.insert(step=step, current_obs=current_obs, action=one_hot_action, action_log_prob=action_log_prob.data,
                           value_pred=value.data, reward=reward, mask=masks)

        # 注意不要引用上述for循环定义的变量。下面变量的命名和使用都要注意。
        next_inp = Variable(rollout.observations[-1], volatile=True)  # 禁止梯度更新
        next_value = actor_critic(next_inp)[0].data  # 获取下一步的value值
        rollout.compute_returns(next_value=next_value, use_gae=False, gamma=args.gamma, tau=None)

        # 下面进行A2C算法梯度更新
        inps = Variable(rollout.observations[:-1].view(-1, *obs_shape))
        acts = rollout.actions.view(-1, num_cls)
        # 下面两步将acts从one_hot变成list
        _, acts = acts.max(1)
        acts = Variable(acts.view(-1, 1))

        # print("a2cs's acts size is {}".format(acts.size()))
        value, action_log_probs, cls_entropy = actor_critic.evaluate_actions(inputs=inps, actions=acts)

        # print("inputs' shape is {}".format(inps.size()))
        # print("value's shape is {}".format(value.size()))
        value = value.view(args.num_steps, args.workers, 1)
        # print("action_log_probs's shape is {}".format(action_log_probs.size()))
        action_log_probs = action_log_probs.view(args.num_steps, args.workers, 1)
        # 计算loss
        advantages = Variable(rollout.returns[:-1]) - value
        value_loss = advantages.pow(2).mean()  # L2Loss or MSE Loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        total_loss = value_loss * args.value_loss_coef + action_loss - cls_entropy * args.entropy_coef

        optimizer.zero_grad()
        total_loss.backward()
        # 下面进行迷之操作。。梯度裁剪（https://www.cnblogs.com/lindaxin/p/7998196.html）
        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
        optimizer.step()

        # 事后一支烟
        rollout.after_update()

        # 存储模型
        if updata_i % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, 'a2c')
            if os.path.exists(save_path) and os.path.isdir(save_path):
                pass
            else:
                os.makedirs(save_path)
            save_file = os.path.join(save_path, str(updata_i)+'.tar')
            save_content = {
                'update_i': updata_i,
                'state_dict': actor_critic.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mean_reward': final_rewards.mean()
            }
            torch.save(save_content, save_file)

        # 输出日志
        if updata_i % args.log_interval == 0:
            end = time.time()
            total_num_steps = (updata_i+1) * args.workers * args.num_steps

            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(updata_i, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), cls_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
            raise NotImplementedError






    envs.close()


def update_current_obs(current_obs, obs):
    """
    全部更新当前的变量（不太明白源代码中为什么这么写？可能是跟fps有关吧，不能保证num_stack为抓取间隔）
    :param current_obs: 当前的observation
    :param obs: 要更新的observation
    """
    obs = torch.from_numpy(obs).float()
    current_obs[:, -args.wave_num:] = obs


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