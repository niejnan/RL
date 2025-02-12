import torch
import gym
import random
import numpy as np
from tqdm import tqdm
from double_dqn import DQN
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env,
                                                   agent.action_dim)
                    next_state, reward, done, trauncated, _ = env.step([action_continuous])

                    done = done or trauncated

                    tup = (state, action, reward, next_state, done)
                    replay_buffer.push(tup)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.get_size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.random_sample(
                            batch_size)
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
    return return_list, max_q_value_list

def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space.low[0]  # 连续动作的最小值
    action_upbound = env.action_space.high[0]  # 连续动作的最大值
    return action_lowbound + (discrete_action /(action_dim - 1)) * (action_upbound - action_lowbound)

def main():
    lr = 1e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 11  # 将连续动作分成11个离散动作

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update)
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                            replay_buffer, minimal_size,
                                            batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(env_name))
    plt.show()


if __name__ == "__main__":
    main()