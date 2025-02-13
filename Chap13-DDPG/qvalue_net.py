import torch
import torch.nn as nn

class QValueNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):

        super(QValueNet, self).__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # s_dim + a_dim
        self.fc1 = nn.Linear(self.state_dim + action_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        """
        输入状态 s 和 动作 a
        """
        cat = torch.cat([x, a], dim=1)
        x = self.relu(self.fc1(cat))

        x = self.relu(self.fc2(x))

        return self.fc_out(x)