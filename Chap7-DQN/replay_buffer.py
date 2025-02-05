import collections
import random
import numpy as np

class ReplayBuffer:

    def __init__(self, size):

        self.buffer = collections.deque(maxlen = size)
    
    def push(self, state, action, reward, next_state, done):

        tup = (state, action, reward, next_state, done)
        self.buffer.append(tup)

    def random_sample(self, batch_size):
        
        # 从 buffer 中随机采样
        transitions = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*transitions)

        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def get_size(self):

        return len(self.buffer)
    