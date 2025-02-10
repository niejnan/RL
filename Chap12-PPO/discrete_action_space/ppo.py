import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

from policy_net import PolicyNet
from value_net import ValueNet
from tqdm import tqdm

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


class PPO:
    """
    PPO 和 TRPO 的区别在于：
    PPO 可以直接用 Adam 或者 SGD 更新
    TRPO 使用二阶优化求解更新

    PPO 的约束通过裁剪近似限制更新幅度
    TRPO 则是通过 KL 散度约束计算最优步长

    PPO 是一阶优化 + 裁剪约束， 用一个简单的 无约束优化 近似 TRPO 复杂的约束优化, 直接梯度下降求解
    TRPO 是二阶优化 + KL 散度约束 + KKT 条件， 本质上是一个 带有不等式约束的凸优化问题
    """
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, epsilon, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)

        self.critic = ValueNet(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = actor_lr)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = critic_lr)

        self.lmbda = lmbda
        self.epochs = epochs
        self.epsilon = epsilon
        self.gamma = gamma

    def take_action(self, state):
        """
        return:
        action(int): 选一个离散的动作
        """
        state = torch.tensor([state], dtype=torch.float)

        action = torch.distributions.Categorical(self.actor(state)).sample()

        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # Q(s,a) = r + γ * V(s')
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        
        # A(s,a) = TD_target - V(s)
        td_delta = td_target - self.critic(states)

        # 通过 GAE 计算优势函数
        a_gae = advantage_GAE(self.gamma, self.lmbda, td_delta.cpu())

        # 旧策略的 log 概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # print(f"state is : {states}")
        # print(f"state.shape is : {states.shape}")
        # print(f"ouput_shape is : {self.actor(states).shape}")
        # print(f"old_log_probs is : {old_log_probs}")

        for _ in range(self.epochs):
            # 新策略的 log 概率
            log_probs = torch.log(self.actor(states).gather(1, actions))

            # 新旧策略重要性采样的比例 π_θ(a|s) / π_θ_old(a|s)
            ratio = torch.exp(log_probs - old_log_probs)

            # 比值 * 优势函数
            surr1 = ratio * a_gae
            # 裁剪, 防止更新幅度过大
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * a_gae

            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


if __name__ == "__main__":
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2

    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma)
    
    return_list = []

    i = 0
    while i == 0:
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state, _ = env.reset(seed = 0)
        print(f"state.shape is : {state.shape}")
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, truncated, _ = env.step(action)

            done = done or truncated

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
            return_list.append(episode_return)
        agent.update(transition_dict)
        i += 1