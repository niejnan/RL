import numpy as np
from sarsa import Sarsa
from cliffwalking import CliffWalking
from tqdm import tqdm
import matplotlib.pyplot as plt

def print_agent(agent, env, action_meaning, disaster = [], end = []):

    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:

                a = agent.best_action(i * env.ncol + j)

                pi_str = ''

                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

if __name__ == "__main__":
    
    env = CliffWalking(12, 4)
    np.random.seed(2025)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    # 初始化 Sarsa
    agent = Sarsa(12, 4, epsilon, alpha, gamma)
    episodes = 500
    return_list = []

    for i in range(10):
        with tqdm(total=int(episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(episodes / 10)):  # 每个进度条的序列数

                episode_return = 0

                state = env.reset()

                action = agent.take_action(state)

                done = False

                while not done:
                    next_state, reward, done = env.step(action)

                    next_action = agent.take_action(next_state)

                    # 这里回报的计算不进行折扣因子衰减
                    episode_return += reward

                    # saras 更新
                    agent.update(state, action, reward, next_state, next_action)

                    # 更新状态和动作
                    state = next_state
                    action = next_action

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Sarsa on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print("Saras 策略")
    print_agent(agent, env, action_meaning, disaster = list(range(37, 47)), end = [47])