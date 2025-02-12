import torch.nn as nn
import torch

class PolicyNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, action_dim)

        self.action_bound = action_bound

    def forward(self, x):
        
        # tanh 到[-1, 1], 乘上一个缩放倍数, 映射到真实值
        return torch.tanh(self.fc2(self.relu(self.fc1(x)))) * self.action_bound