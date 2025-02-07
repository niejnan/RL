import gym
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from tqdm import tqdm
from dqn import DQN
from utils import moving_average

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_pictrue(return_list, env_name):
    episodes_list = []
    for i in range(len(return_list)):
        episodes_list.append(i)
    
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
    

def main(lr, num_episodes, hidden_dim, gamma, epsilon, target_update_freq, buffer_size, minimal_size, batch_size, seed):
    env_name = 'CartPole-v0'

    env = gym.make('CartPole-v0')

    set_seed(seed)

    replay_buffer = ReplayBuffer(buffer_size)

    state_dim = env.observation_space.shape[0]
    print(f"state_dim is {type(state_dim)}")

    action_dim = env.action_space.n
    print(f"action_dim is {type(action_dim)}")

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update_freq)

    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:

            for episode in range(int(num_episodes / 10)):

                episode_return = 0

                state, _ = env.reset()
                # print(f"state is {state.shape}")
                
                done = False

                while done is not True:

                    action = agent.take_action(state)

                    # 采取动作后 返回 下一个状态、奖励、是否结束的标志
                    next_state, reward, done, truncated, _ = env.step(action)

                    done = done or truncated

                    # 将五元组 (state, action, reward, next_state, done) 加入 buffer
                    push_tuple = (state, action, reward, next_state, done)
                    replay_buffer.push(push_tuple)

                    # 更新到下一个状态
                    state = next_state

                    episode_return += reward

                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.get_size() > minimal_size:

                        buffer_states, buffer_actions, buffer_rewards, buffer_next_states, buffer_dones = replay_buffer.random_sample(batch_size)

                        transition_dict = {
                            'states': buffer_states,
                            'actions': buffer_actions,
                            'next_states': buffer_next_states,
                            'rewards': buffer_rewards,
                            'dones': buffer_dones
                        }

                        agent.update(transition_dict)

                return_list.append(episode_return)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    get_pictrue(return_list, env_name)

if __name__ == "__main__":
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000

    # 超过 500 才用经验回放
    minimal_size = 500
    batch_size = 64

    seed = 0
    main(lr, num_episodes, hidden_dim, gamma, epsilon, target_update, buffer_size, minimal_size, batch_size, seed)










