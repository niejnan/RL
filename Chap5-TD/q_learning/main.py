from cliffwalking import CliffWalking
from q_learning import QLearning
import numpy as np
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

    np.random.seed(2025)
    env = CliffWalking(12, 4)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = QLearning(12, 4, epsilon, alpha, gamma)
    num_episodes = 500

    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0

                state = env.reset()

                done = False

                while not done:
                    action = agent.take_action(state)

                    next_state, reward, done = env.step(action)

                    episode_return += reward

                    agent.update(state, action, reward, next_state)

                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
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
    plt.title('Q-learning on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print('Q-learning算法')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])