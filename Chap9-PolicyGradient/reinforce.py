from policy_net import PolicyNet
import torch.optim as optim
import torch


class REINFORCE:

    def __init__(self, state_dim, hidden_dim, action_dim, lr, gamma):

        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.gamma = gamma

    def take_action(self, state):

        state = torch.tensor([state], dtype=torch.float)

        output = self.policy_net(state)

        action_list = torch.distributions.Categorical(output)

        action = action_list.sample()

        return action.item()
    
    def update(self, transition_dict):

        rewards = transition_dict['rewards']
        states = transition_dict['states']
        actions = transition_dict['actions']

        G = 0

        self.optimizer.zero_grad()

        # 从后向前计算 reward
        for i in reversed(range((len(rewards)))):
            reward = rewards[i]
            
            state = torch.tensor([states[i]], dtype=torch.float)

            action = torch.tensor([actions[i]]).view(-1, 1)

            log_prob = torch.log(self.policy_net(state).gather(1, action))

            G = self.gamma * G +reward
            
            loss = -log_prob * G

            loss.backward()
        
        self.optimizer.step()