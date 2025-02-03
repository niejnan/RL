from cliffwalking import CliffWalking
from utils import print_agent

class ValueIteration:
    """
    价值迭代
    """
    def __init__(self, env, theta, gamma):
        """
        ncol: env 的列数
        nrow: env 的行数
        state_nums: 状态数目
        """
        self.env = env
        self.ncol = self.env.ncol
        self.nrow = self.env.nrow
        self.state_nums = self.ncol * self.nrow
        self.v = [0] * self.state_nums

        self.theta = theta
        self.gamma = gamma
        
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]
    
    def compute_qsa(self, state, action):
        """
        计算 Q(s,a)
        """
        qsa = 0
        # 终点 done = 1, 不计算 V, 只需要加上 r 就好
        for p, next_state, r, done in self.env.P[state][action]:
            qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
        return qsa
    
    def value_iter(self):
        cnt = 0

        while 1:
            max_diff = 0
            new_v = [0] * self.state_nums
            for state in range(self.state_nums):
                qsa_list = []
                for action in range(4):
                    qsa_list.append(self.compute_qsa(state, action))
                
                new_v[state] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[state] - self.v[state]))

            self.v = new_v

            if max_diff < self.theta: break
            cnt += 1
        print(f"价值迭代一共 {cnt} 轮")
        self.get_policy()

    def get_policy(self):

        for state in range(self.state_nums):
            qsa_list = []
            for action in range(4):
                qsa_list.append(self.compute_qsa(state, action))
            
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)

            self.pi[state] = [1 / cntq if q == maxq else 0 for q in qsa_list]

if __name__ == "__main__":
    env = CliffWalking()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iter()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])

                
