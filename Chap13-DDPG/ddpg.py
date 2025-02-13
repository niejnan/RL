
import torch
import torch.optim as optim

from policy_net import PolicyNet
from qvalue_net import QValueNet

import numpy as np

class DDPG:

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim

        self.mse_loss = torch.nn.MSELoss()

    def take_action(self, state):

        state = torch.tensor([state], dtype=torch.float)

        action = self.actor(state).item()

        # 添加噪声，增加探索
        return action + self.sigma * np.random.randn(self.action_dim)
    
    def soft_update(self, current_net, target_net):

        target_params = list(target_net.parameters())
        current_params = list(current_net.parameters())

        for i in range(len(target_params)):
            para_target = target_params[i]
            para = current_params[i]

            # 软更新
            para_target.data = para_target.data * (1.0 - self.tau) + para.data * self.tau

    def update(self, transition_dict):

        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        next_q = self.target_critic(next_states, self.target_actor(next_states))

        q_target = rewards + self.gamma * next_q * (1 - dones)

        critic_loss = torch.mean(self.mse_loss(self.critic(states, actions), q_target))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)

        self.soft_update(self.critic, self.target_critic)



