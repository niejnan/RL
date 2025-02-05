from cliffwalking import CliffWalking
from tqdm import tqdm
from dyna_q import Dyna_Q
import numpy as np
import random
import matplotlib.pyplot as plt
import time

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

def main(n_planing, num_episodes):

    env = CliffWalking(12, 4)

    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9

    agent = Dyna_Q(12, 4, epsilon, alpha, gamma, n_planing)

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

                    # 更新 state
                    state = next_state
                
                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': i_episode + 1,
                        'return': np.mean(return_list[-10:])
                    })
                    pbar.update(1)
    
    return return_list

if __name__ == "__main__":

    nums_episodes = 300

    np.random.seed(0)
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)

        return_list = main(n_planning, nums_episodes)

        episodes_list = list(range(len(return_list)))

        plt.plot(episodes_list, return_list, label=str(n_planning) + ' planning steps')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()