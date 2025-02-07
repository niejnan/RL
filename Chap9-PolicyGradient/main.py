import gym
import numpy as np
import random
import torch
from reinforce import REINFORCE
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import moving_average

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

def main(lr, num_episodes, hidden_dim, gamma):

    set_seed(seed)

    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, hidden_dim, action_dim, lr, gamma)

    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for episode in range(int(num_episodes / 10)):
                # 当前轨迹的 return
                episode_return = 0
                # 当前轨迹的 dict
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}

                state, _ = env.reset()

                done = False

                # 当还没有到终点的时候
                while done is not True:
                    #  传入状态, 选择动作
                    action = agent.take_action(state=state)

                    # 执行动作, 得到 下一个状态、 奖励、 是否完成标志
                    next_state, reward, done, truncated, _ = env.step(action)
                    
                    done = done or truncated

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state

                    episode_return += reward
                return_list.append(episode_return)

                agent.update(transition_dict)

                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    get_picture(return_list, env_name)


if __name__ == "__main__":
    lr = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98

    seed = 2025

    main(lr, num_episodes, hidden_dim, gamma)