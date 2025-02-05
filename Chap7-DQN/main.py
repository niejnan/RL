import gym
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from tqdm import tqdm
from dqn import DQN
from utils import moving_average

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main (lr, num_episodes, hidden_dim, gamma, epsilon, target_update, buffer_size, minimal_size, batch_size, seed):
    env_name = 'CartPole-v0'
    env = gym.make('CartPole-v0')
    set_seed(seed)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False

                while not done:
                    action = agent.take_action(state)

                    next_state, reward, done, truncated, info = env.step(action)

                    replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.get_size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.random_sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)

    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

if __name__ == "__main__":
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64

    seed = 0
    main(lr, num_episodes, hidden_dim, gamma, epsilon, target_update, buffer_size, minimal_size, batch_size, seed)
    
