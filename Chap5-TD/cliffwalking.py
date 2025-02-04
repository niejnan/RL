"""
与 chap4 环境不同，不需要奖励函数和状态转移函数
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalking:

    def __init__(self, ncol, nrow):
        self.ncol = ncol
        self.nrow = nrow

        self.x = 0
        self.y = self.nrow - 1

    def step(self, action):

        directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]

        self.x = min(self.ncol - 1, max(0, self.x + directions[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + directions[action][1]))

        next_state = self.y * self.ncol + self.x # 二维空间的位置，转化为一维的索引，通过数组索引
        reward = -1

        done = False

        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            
            if self.x != self.ncol - 1:
                reward = -100
        
        return next_state, reward, done

    def reset(self):

        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x