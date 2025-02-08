import gym
import matplotlib.pyplot as plt
import random
import torch
import numpy as np

from tqdm import tqdm
from utils import moving_average
from actor_critic import ActorCritic


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_picture(return_list, env_name):
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('REINFORCE on {}'.format(env_name))
    plt.show()

def main(actor_lr, critic_lr, num_episodes, hidden_dim, gamma = 0.98):
    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma)

    return_list = []
    # on-policy
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)

                    next_state, reward, done, truncated, _ = env.step(action)

                    done = done or truncated

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state

                    episode_return += reward

                # 一个 episode 结束后, 才添加 reward 
                return_list.append(episode_return)
                # 
                agent.update(transition_dict)

                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    
    get_picture(return_list, env_name)

if __name__ == "__main__":
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    main(actor_lr, critic_lr, num_episodes, hidden_dim, gamma)
