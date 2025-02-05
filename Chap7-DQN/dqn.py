import torch
from q_net import QNet
from replay_buffer import ReplayBuffer
import torch.nn as nn
import numpy as np

class DQN:
    """
    target_update: 目标网络更新频率
    """
    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update):

        self.action_dim = action_dim
        self.q_net = QNet(state_dim, hidden_dim, self.action_dim)

        self.target_q = QNet(state_dim, hidden_dim, self.action_dim)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.loss_func = nn.MSELoss()

        self.gamma = gamma

        self.epsilon = epsilon

        self.target_update = target_update
        self.cnt = 0
    
    def take_action(self, state):

        if np.random.random() < self.epsilon:

            action = np.random.randint(self.action_dim)
        else:
            
            # 转 tensor
            state = torch.tensor([state], dtype=torch.float)

            # forward = self.q_net(state)
            
            # action = torch.argmax(forward).item()

            action = self.q_net(state).argmax().item()

        
        return action
    
    def update(self, transition_dict):

        # 从 transition_dict 中获取状态、动作、奖励、下一状态和是否结束的标志
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        # actions 转化成列向量
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        output = self.q_net(states)
        # print(f"states 前向传播的形状是 {output.shape}")

        q = output.gather(1, actions)
        # print(f"q 的形状是 {q.shape}")

        next_q =  self.target_q(next_states).max(1)[0].view(-1, 1)
        # print(f"next_q 的形状是 {next_q.shape}")

        except_q = rewards + self.gamma * next_q * (1 - dones)
        # print(f"except_q 的形状是 {except_q.shape}")

        # 当前 Q 值和目标 Q 值的差异
        loss = self.loss_func(q, except_q)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.cnt % self.target_update == 0:
            # 用加载权重的方式完成更新
            self.target_q.load_state_dict(self.q_net.state_dict())

        self.cnt += 1






