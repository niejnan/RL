import numpy as np

class Sarsa:
    # action_nums = 4
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, action_nums = 4):
        
        self.action_nums = action_nums
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q_table = np.zeros((nrow * ncol, action_nums))

    def take_action(self, state):

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_nums)
        else:
            action = np.argmax(self.Q_table[state])
        
        return action
    
    def best_action(self, state):
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.action_nums)]

        for index in range(self.action_nums): # 遍历记录价值一样的动作
            if self.Q_table[state, index]== Q_max:
                a[index] = 1
        return a
    
    # s,a,r,s,a
    def update(self, s0, a0, r, s1, a1):
        delta_t = r + self.gamma * self.Q_table[s1][a1] - self.Q_table[s0][a0]

        self.Q_table[s0][a0] += self.alpha * delta_t