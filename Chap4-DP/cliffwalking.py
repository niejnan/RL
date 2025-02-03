
class CliffWalking:
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
                for action in range(4):
                    if i == self.nrow - 1 and j > 0: # 判断是否到了悬崖
                        P[i * self.ncol + j][action] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 若不在悬崖,计算下一个状态
                    delta_x, delta_y = directions[action]
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

                    P[i * self.ncol + j][action] = [(1, next_state, reward, done)] # 确定性环境，全部都是 1
        return P
