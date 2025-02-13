import torch
import random
import numpy as np
import gym
import matplotlib.pyplot as plt

from sac import SAC
from replay_buffer import ReplayBuffer

from tqdm import tqdm


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                # 当游戏未结束时
                while not done:
                    # 1. 输入状态 s, 采取动作 a
                    action = agent.take_action(state)
                    # 2. 得到环境的反馈 rewards, 以及下一个状态 next_state
                    next_state, reward, done, truncated, _ = env.step(action)

                    done = done or truncated
                    tup = (state, action, reward, next_state, done)
                    replay_buffer.push(tup)
                    # 3. 更新状态
                    state = next_state
                    # 4. 累加奖励, 这里不需要 * gamma 因为算的是累积的奖励, 用来画图的
                    episode_return += reward

                    # 5. 若 buffer 大于一定数量了, 可以取一部分数据出来训练
                    if replay_buffer.get_size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.random_sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}

                        # 6. 输入 buffer 里的经验数据, 用于训练模型, 更新模型的策略
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list



def main():
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    replay_buffer = ReplayBuffer(buffer_size)
    agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
                        actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                        gamma)

    return_list = train_off_policy_agent(env, agent, num_episodes,
                                                replay_buffer, minimal_size,
                                                batch_size)
    
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('SAC on {}'.format(env_name))
    plt.show()

if __name__ == "__main__":
    main()