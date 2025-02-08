import torch.nn as nn
import torch

class PolicyNet(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        return self.softmax(self.fc2(self.relu(self.fc1(x))))
    
if __name__ == "__main__":

    model = PolicyNet(20, 10, 2)
    x = torch.randn(1, 20)

    output = model(x)

    print(output)
    print(output.shape)