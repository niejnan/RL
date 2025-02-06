
import collections
import random
import numpy as np
from typing import List, Tuple

class ReplayBuffer:
    """
    size: buffer 的大小
    双端队列实现 buffer
    队列中的元素为五元组 (state, action, reward, next_state, done)
    """

    def __init__(self, size):
        
        # 用双端队列实现 buffer
        self.buffer = collections.deque(maxlen = size)
    
    def push(self, tup):    
        # 将五元组 (state, action, reward, next_state, done) 加入 buffer
        self.buffer.append(tup)

    def random_sample(self, batch_size):

        # 从 buffer 中随机采样
        transitions : List[Tuple] = random.sample(self.buffer, batch_size)

        # 将五元组拆分成五个 tuple
        states, actions, rewards, next_states, dones = zip(*transitions)

        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def get_size(self):
        
        return len(self.buffer)
    

if __name__ == "__main__":
    buffer = ReplayBuffer(10)
    for i in range(20):
        buffer.push(i, i, i, i, i)
    print(buffer.random_sample(5))
    