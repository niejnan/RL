import torch.nn as nn
import torch

class QNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, action_dim):

        super(QNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):

        x = self.fc1(x)

        x = self.relu(x)
        x = self.fc2(x)

        return x

    def save(self, path):

        torch.save(self.state_dict(), path)
    
