from policy_net import PolicyNet
from value_net import ValueNet
import torch
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic:

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma):


        self.actor = PolicyNet(state_dim, hidden_dim, action_dim)

        self.critic = ValueNet(state_dim, hidden_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma

    def take_action(self, state):

        state = torch.tensor([state], dtype=torch.float)

        prob = self.actor(state)

        action_list = torch.distributions.Categorical(prob)

        action = action_list.sample()

        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)

        actions = torch.tensor(transition_dict['actions']).view(-1, 1)

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)

        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)

        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # V(s_t)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)

        td_delta = td_target - self.critic(states)

        log_probs = torch.log(self.actor(states).gather(1, actions))

        # actor 来源于 策略梯度，用策略梯度的方法优化 策略
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        # 让 Critic 取拟合 状态价值函数 V(s)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()