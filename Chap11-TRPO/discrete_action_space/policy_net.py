import torch.nn as nn
import torch
import torch.nn.functional as F

"""
同 Actor-Critic
"""

# class PolicyNet(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.relu = nn.ReLU()

#         self.fc2 = nn.Linear(hidden_dim, action_dim)
#         self.softmax = nn.Softmax()

#     def forward(self, x):
#         return self.softmax(self.fc2(self.relu(self.fc1(x))))


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
