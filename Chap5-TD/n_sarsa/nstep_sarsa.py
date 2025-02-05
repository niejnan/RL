import numpy as np


class nstep_Sarsa:
    """
    n: n-step Sarsa 的步数, 考虑多少步的回报
    """
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, action_nums = 4):
        self.action_nums = action_nums
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n = n

        self.Q_table = np.zeros((nrow * ncol, action_nums))

        # 暂存 n 步的状态, 动作, 奖励, 相比于单步的 sarsa 多了一个 n 步的状态
        self.state_list = []
        self.action_list = []
        self.reward_list = []
    
    def take_action(self, state):
        """
        epsilon-greedy
        """
        # 生成 [0.0, 1.0) 的随机 float
        if np.random.random() < self.epsilon:
            # 生成 [0, action_nums) 的随机 int
            action = np.random.randint(self.action_nums)
        else:
            action = np.argmax(self.Q_table[state])
        
        return action
    
    def best_action(self, state):
        """
        提取最大价值的 action
        """
        
        # 从 Q_table 中取出最大值
        Q_max = np.max(self.Q_table[state])

        # 用于记录价值最大的动作
        a = [0 for _ in range(self.action_nums)]

        # 看看还有没有相同价值的动作, 标记为 1
        for index in range(self.action_nums):
            if self.Q_table[state, index] == Q_max:
                a[index] = 1
        return a

    def update(self, s0, a0, r, s1, a1, done):

        # 存储 n 步的状态, 动作, 奖励
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)

        # 如果长度超过 n, 就开始更新
        if len(self.reward_list) == self.n:
            
            # Q(s_{t+n}, a_{t+n})
            G = self.Q_table[s1, a1]

            # 从后往前 一个一个算
            for index in reversed(range(self.n)):
                G = self.reward_list[index] + self.gamma * G

                if done and index > 0:
                    # 如果是终点, 且不是最后一个状态, 就不更新
                    s = self.state_list[index]

                    a = self.action_list[index]

                    # 更新 Q
                    self.Q_table[s,a] += self.alpha * (G - self.Q_table[s,a])
            
            # 删除暂存的状态
            s = self.state_list.pop(0)

            a = self.action_list.pop(0)

            self.reward_list.pop(0)

            self.Q_table[s, a] = self.alpha * (G - self.Q_table[s, a])
        else:
            """
            如果没有完成, 就不更新, 跳出 if
            """

        # 若完成，清除缓存
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []

class nstep_Sarsa:
    """ n步Sarsa算法 """
    def __init__(self, n, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n  # 采用n步Sarsa算法
        self.state_list = []  # 保存之前的状态
        self.action_list = []  # 保存之前的动作
        self.reward_list = []  # 保存之前的奖励

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1, done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:  # 若保存的数据可以进行n步更新
            G = self.Q_table[s1, a1]  # 得到Q(s_{t+n}, a_{t+n})
            for i in reversed(range(self.n)):
                G = self.gamma * G + self.reward_list[i]  # 不断向前计算每一步的回报
                # 如果到达终止状态,最后几步虽然长度不够n步,也将其进行更新
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.state_list.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            # n步Sarsa的主要更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:  # 如果到达终止状态,即将开始下一条序列,则将列表全清空
            self.state_list = []
            self.action_list = []
            self.reward_list = []
    