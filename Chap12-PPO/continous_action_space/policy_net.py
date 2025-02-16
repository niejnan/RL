import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    

class PolicyNetContinuous(nn.Module):
    """
    连续动作空间
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x))

        return mu, std
    
    