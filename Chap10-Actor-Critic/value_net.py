import torch.nn as nn

class ValueNet(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        return self.fc2(self.relu(self.fc1(x)))
    



