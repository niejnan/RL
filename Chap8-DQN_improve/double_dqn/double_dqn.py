import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

from q_net import QNet

class DQN:

    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, dqn_type = 'VanillaDQN'):
        
        self.action_dim = action_dim

        self.q_net = QNet(state_dim, hidden_dim, self.action_dim)

        self.target_q_net = QNet(state_dim, hidden_dim, self.action_dim)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma

        self.epsilon = epsilon

        self.target_update = target_update

        self.cnt = 0

        self.dqn_type = dqn_type

        self.mse_loss = nn.MSELoss()

    def take_action(self, state):
        """
        epsilon-贪婪
        """
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float)
            # 根据 QNet 选动作
            action = self.q_net(state).argmax().item()
        return action
    

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float)
        
        return self.q_net(state).max().item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions'])
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # print("states shape:", states.shape)
        # print("actions shape:", actions.shape)
        # print("q_net(states) shape:", self.q_net(states).shape)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)) 
        # 下个状态的最大Q值

        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            
            # 
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(self.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.cnt % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.cnt += 1


