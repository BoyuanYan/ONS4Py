import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage(object):

    def __init__(self, num_steps: int, num_processes: int, obs_shape: tuple, action_shape: int):
        """

        :param num_steps: 进行一次训练所需要游戏进行的步骤数
        :param num_processes: 同时运行的游戏进程数
        :param obs_shape: observation space的shape
        :param action_shape: action space的size
        """
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, action_shape).long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)  # 游戏是否结束的标记

    def cuda(self):
        self.observations = self.observations.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()

    def insert(self, step, current_obs, action, action_log_prob, value_pred, reward, mask):
        self.observations[step + 1].copy_(current_obs)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau=None):
        """
        计算returns。在不考虑use_gae,tau的情况下，递推公式如下：
        $$return_i = return_{i+1} \cdot \gamma \cdot mask_{i+1} + reward_i$$
        :param next_value: 下一个值，use_gae为False的时候，表示下一个return。
        :param use_gae: 相关解释，可见https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/issues/49
        :param gamma: 计算reward的时候，评估当前选择对之后状态变化的影响的折扣因子
        :param tau: gae模式下用到的参数，不懂
        :return: 计算returns值
        """
        if use_gae:
            # 把最后一个值改成next_value
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            # 把returns第一个维度的末尾值对应的矩阵，全部改成next_value的值
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]