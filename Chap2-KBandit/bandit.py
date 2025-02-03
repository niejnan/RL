import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:

    def __init__(self, k):

        self.probs = np.random.uniform(size = k)

        self.best_idx = np.argmax(self.probs)
        self.best_prob = self.probs[self.best_idx]

        self.k = k

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

if __name__ == "__main__":
    np.random.seed(2025)

    k = 10

    bandit = BernoulliBandit(k)

    print(f"{k} 臂老虎机")
    print(f"获奖概率最大的拉杆为{bandit.best_idx}号,其获奖概率为{bandit.best_prob:.4f}")
