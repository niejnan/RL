import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as py
import torch.nn.functional as F

from policy_net import PolicyNetContinuous
from value_net import ValueNet


def advantage_GAE(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()

    advantage_list = []
    advantage = 0.0

    # 逆序遍历
    for delta in reversed(td_delta):
        # A_t = δ_t + (γλ) * A_{t+1}
        advantage = delta + gamma * lmbda * advantage

        advantage_list.append(advantage)

    advantage_list.reverse()
    
    return torch.tensor(advantage_list, dtype=torch.float)

class PPOContinuous:
    """
    策略网络输出连续动作高斯分布的 mean 和 std
    连续动作在高斯分布中采样

    
    PPO 和 TRPO 的区别在于：
    PPO 可以直接用 Adam 或者 SGD 更新
    TRPO 使用二阶优化求解更新

    PPO 的约束通过裁剪近似限制更新幅度
    TRPO 则是通过 KL 散度约束计算最优步长

    PPO 是一阶优化 + 裁剪约束， 用一个简单的 无约束优化 近似 TRPO 复杂的约束优化, 直接梯度下降求解
    TRPO 是二阶优化 + KL 散度约束 + KKT 条件， 本质上是一个 带有不等式约束的凸优化问题
    """

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, epislon, gamma):

        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim)

        self.critic = ValueNet(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr= critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.epsilon = epislon

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        mu, sigma = self.actor(state)

        action_list = torch.distributions.Normal(mu, sigma)
        action = action_list.sample()

        return [action.item()]
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1)
        
        rewards = (rewards + 8.0) / 8.0

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_delta = td_target - self.critic(states)

        a_gae = advantage_GAE(self.gamma, self.lmbda, td_delta.cpu())

        mu, std = self.actor(states)

        action_list = torch.distributions.Normal(mu.detach(), std.detach())

        old_log_probs = action_list.log_prob(actions)

        for _ in range(self.epochs):

            mu, std = self.actor(states)

            action_list = torch.distributions.Normal(mu, std)

            log_probs = action_list.log_prob(actions)

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * a_gae

            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * a_gae

            actor_loss = torch.mean(-torch.min(surr1, surr2))

            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


    
