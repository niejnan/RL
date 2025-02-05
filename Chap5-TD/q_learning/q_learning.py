import numpy as np

class QLearning:

    def __init__(self, nrow, ncol, epsilon, alpha, gamma, action_nums = 4):
        
        # 悬崖环境 aciton_nums = 4
        self.action_nums = action_nums

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.nrow = nrow
        self.ncol = ncol

        # 行 nrow * ncols, 列 action_nums
        self.Q_table = np.zeros((nrow * ncol, action_nums))

    def take_action(self, state):
        """
        选取下一步的动作
        """

        if np.random.random() < self.epsilon:

            action = np.random.randint(self.action_nums)
        else:

            action = np.argmax(self.Q_table[state])
        
        return action
    
    def best_action(self, state):

        Q_max = np.max(self.Q_table[state])
        
        a = [0 for i in range(self.action_nums)]

        for i in range(self.action_nums):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        
        return a

    def update(self, s0, a0, r, s1):
        
        # q_learning 的TD 误差
        # 选 s1 状态下最大的 Q 值, 用于更新
        delta_t = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0][a0]

        self.Q_table[s0, a0] += self.alpha * delta_t