import torch.nn as nn
import torch

class PolicyNetContinuous(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):

        super(PolicyNetContinuous, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.fc_mu = nn.Linear(hidden_dim, action_dim)

        self.fc_std = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()

        self.softplus = nn.Softplus()

        self.action_bound = action_bound
    
    def forward(self, x):

        x = self.relu(self.fc1(x))

        mu = self.fc_mu(x)

        std = self.softplus(self.fc_std(x))

        # 正态分布，参数为均值 (mu) 和标准差 (std)
        dist = torch.distributions.Normal(mu, std)

        # 从正态分布中采样动作
        normal_sample = dist.rsample()

        # 计算动作的对数概率（log probability）
        log_prob = dist.log_prob(normal_sample)

        # 映射
        action = torch.tanh(normal_sample)

        # tanh 的导数会影响 log_porb ,调整 log_prob
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)

        action = action * self.action_bound

        return action, log_prob
    