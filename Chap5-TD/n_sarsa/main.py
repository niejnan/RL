import numpy as np  
import matplotlib.pyplot as plt
from tqdm import tqdm
from nstep_sarsa import nstep_Sarsa
from cliffwalking import CliffWalking

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

"""
main 的代码和 Sarsa.main 基本一致, 只是更新时需要考虑 n 步的回报
"""

if __name__ == "__main__":
    np.random.seed(0)
    env = CliffWalking(12, 4)

    # 5步Sarsa
    n_step = 5

    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    agent = nstep_Sarsa(n_step, 12, 4, epsilon, alpha, gamma)
    num_episodes = 500

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False

                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)

                    episode_return += reward

                    agent.update(state, action, reward, next_state, next_action, done)

                    state = next_state
                    action = next_action

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
    plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print("nstep-Saras 策略")
    print_agent(agent, env, action_meaning, disaster = list(range(37, 47)), end = [47])