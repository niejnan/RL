import torch
import torch.nn as nn
import numpy as np
from q_net import QNet

class DQN:
    """
    target_update_freq: 目标网络更新频率
    """
    def __init__(self, state_dim, hiden_dim, action_dim, lr, gamma, epsilon, target_update_freq):

        self.action_dim = action_dim

        # QNet
        self.q_net = QNet(state_dim, hiden_dim, action_dim)

        # 目标 QNet
        self.target_q_net = QNet(state_dim, hiden_dim, action_dim)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.loss_func = nn.MSELoss()

        self.gamma = gamma

        self.epsilon = epsilon

        self.target_update_freq = target_update_freq

        # 计数器 统计更新次数
        self.cnt = 0

    def take_action(self, state):

        if np.random.random() < self.epsilon:

            action = np.random.randint(self.action_dim)

        else:
            
            # 转 tensor
            state = torch.tensor([state], dtype=torch.float)

            ouput = self.q_net(state)

            action = ouput.argmax().item()
        
        return action
    
    def update(self, transition_dict):
        """
        当 buffer 数据超过一定的量以后，可以进行 Q 网络训练
        transition_dict: 从 buffer 中 sample 出来的数据

        每调用 update 一次，更新一次 Q 网络
        每调用 update 函数 target_update_freq 次，就更新一次目标网络
        """

        # 从 dict 中获取状态、动作、奖励、下一个状态和是否结束的标志
        states = torch.tensor(transition_dict['states'], dtype=torch.float)

        # actions, rewards 转化成列向量
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)

        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)

        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # 前向传播
        output = self.q_net(states)
        # print(f"states 前向传播的形状是 {output.shape}")

        q = output.gather(1, actions)
        # print(f"q 的形状是 {q.shape}")

        # 计算目标 Q 值
        next_q = self.target_q_net(next_states).max(dim=1, keepdim=True)[0].view(-1, 1)
        # print(f"next_q 的形状是 {next_q.shape}")

        # 用来更新 Q 值，作为短暂的目标
        except_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_func(q, except_q)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # 每更新 target_update_freq 次，就更新一次目标网络
        if self.cnt % self.target_update_freq == 0:
            # 用加载权重的方式完成更新
            self.target_q_net.load_state_dict(self.q_net.state_dict(), strict=True)
        
        self.cnt += 1


