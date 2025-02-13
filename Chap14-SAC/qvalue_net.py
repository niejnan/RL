import torch
import torch.nn as nn

class QValueContinuous(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueContinuous, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        
        cat = torch.cat([state, action], dim=1)

        x = self.relu(self.fc1(cat))

        x = self.relu(self.fc2(x))

        return self.fc_out(x)