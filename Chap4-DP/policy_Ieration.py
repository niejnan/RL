import copy
from iteration import PolicyIteration

class CliffWalkingEnv:
    """
    模拟了一个 4×12 的网格环境：
	起点 在 (3,0)，即网格左下角。
	终点 在 (3,11)，即网格右下角。
	悬崖 位置是 (3,1) 到 (3,10)，一旦掉下去，奖励为 -100，并回到起点。
	普通移动 每次奖励 -1，网格允许 4 个方向移动（上、下、左、右）。
	存储状态转移规则 到 P[state][action]，用于策略评估。
    """
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 列
        self.nrow = nrow  # 行

        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP() # p = 1 , 确定环境

    def createP(self):
        # 初始化
        P = []
        for state in range(self.nrow * self.ncol):
            actions = []
            for _ in range(4):
                actions.append([])
            P.append(actions)
        
        # 定义动作方向,上下左右
        directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        # [(p, next_state, reward, done)]包含下一个状态和奖励 
        for i in range(self.nrow):
            for j in range(self.ncol):
                for k in range(4):
                    if i == self.nrow - 1 and j > 0: # 判断是否到了悬崖
                        P[i * self.ncol + j][k] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 若不在悬崖,计算下一个状态
                    delta_x, delta_y = directions[k]
                    new_x , new_y = j + delta_x, i + delta_y

                    # 处理边界, 避免越界
                    nextX = max(0, min(self.ncol - 1, new_x))
                    nextY = max(0, min(self.nrow - 1, new_y))

                    next_state = nextY * self.ncol + nextX

                    reward = -1 # 默认 -1
                    done = False # 默认 不是停止状态

                    if nextY == self.nrow - 1 and nextX > 0: # 如果下一个位置在悬崖或是在终点
                        done = True

                        if nextX != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100 # 扣钱

                    P[i * self.ncol + j][k] = [(1, next_state, reward, done)] # 确定性环境，全部都是 1
        return P
    
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


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]),
                  end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

if __name__ == "__main__":
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = PolicyIteration(env, theta, gamma)
    agent.policy_iter()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])
