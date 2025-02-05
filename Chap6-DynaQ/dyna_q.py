import numpy as np
import random

class Dyna_Q:

    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planing, action_nums = 4):

        self.action_nums = action_nums
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.nrow = nrow
        self.ncol = ncol
        self.n_planing = n_planing

        self.Q_table = np.zeros((nrow * ncol, action_nums))

        # 用 dict 存 model
        self.model = {}

    def take_action(self, state):

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_nums)
        else:
            action = np.argmax(self.Q_table[state])

        return action

    def q_learning(self, s0, a0, r, s1):

        delta_t = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]

        self.Q_table[s0, a0] += self.alpha * delta_t
    
    def update(self, s0, a0, r, s1):

        self.q_learning(s0, a0, r, s1)

        # 存 model, 在 s0选择动作 a0，得到 r, s1
        self.model[(s0, a0)] = r, s1

        for _ in range(self.n_planing):

            model_list = list(self.model.items())

            (random_s, random_action), (random_r, random_s_prime) = random.choice(model_list)

            # 学
            self.q_learning(random_s, random_action, random_r, random_s_prime)
