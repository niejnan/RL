import torch.nn as nn

class QNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))