import copy

class PolicyIteration:

    def __init__(self, env, theta, gamma):

        self.env = env

        self.v = [0] * self.env.ncol * self.env.nrow

        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]

        self.theta = theta
        self.gamma = gamma
    
    def compute_qsa(self, state, action):
        """
        计算 Q(s,a)
        """
        qsa = 0
        for p, next_state, r, done in self.env.P[state][action]:
            qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
        return qsa

    def policy_eval(self):
        """
        利用贝尔曼方程计算每个状态的价值
        """
        cnt = 1

        while 1:
            max_diff = 0
            # 状态价值函数 V(s)
            new_v = [0] * self.env.ncol * self.env.nrow
            
            for state in range(self.env.ncol * self.env.nrow):
                # 存储所有的 Q(s,a)
                qsa_list = []
                # 四个动作
                for action in range(4):
                    qsa_list.append(self.pi[state][action] * self.compute_qsa(state, action))
                
                # 计算 V(s)
                new_v[state] = sum(qsa_list)

                # 计算最大误差
                max_diff = max(max_diff, abs(new_v[state] - self.v[state]))
            
            # 更新 V
            self.v = new_v
            # 小于 theta 认为收敛
            if max_diff < self.theta: break
            cnt += 1

        print(f"策略评估了 {cnt} 轮后完成")
    
    def policy_improve(self):

        for state in range(self.env.nrow * self.env.ncol):
            # 存 Q(s,a), qsa_list 存的是当前 state 下 4 个动作的 Q
            qsa_list = []
            for action in range(4):
                qsa_list.append(self.compute_qsa(state, action))

            # 选择最优动作
            maxq = max(qsa_list)
            # 统计有几个动作得到了最大的 Q 
            cntq = qsa_list.count(maxq)

            # 如[0, 0.5, 0.5, 0] 下和左是最优动作，
            self.pi[state] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        
        print("策略提升完成")
        return self.pi
    
    def policy_iter(self):
        while 1:
            self.policy_eval()
            # 深拷贝
            old_pi = copy.deepcopy(self.pi)

            new_pi = self.policy_improve()

            # 旧的策略等于新的策略的时候，停止迭代
            if old_pi == new_pi: break
