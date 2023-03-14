import numpy as np
from random import random, randint
from matplotlib import pyplot as plt

class NormalDistBandit:
    def __init__(self, means, stds):
        assert len(means) == len(stds), "Means and stds must be the same length."
        self.n = len(means)
        self.means = np.array(means)
        self.stds = np.array(stds)
        assert all(self.stds >= 0), "Stds must be positive."
    
    def pull(self, k):
        assert 0 <= k < self.n, f"Invalid arm {k}."
        return np.random.normal(loc=self.means[k], scale=self.stds[k])

def epsilon_greedy(values, epsilon):
    assert len(values) > 1, "There should be 2 or more values."
    eps = epsilon * len(values) / (len(values) - 1)
    if random() <= eps:
        return randint(0, len(values)-1)
    return int(np.argmax(values))
   
        
if __name__ == "__main__":
    n = 5
    bandit = NormalDistBandit(means = np.array(range(-n, n+1)), stds = np.ones(11))
    
    # Task：把下面这几种epsilon的曲线画到一张图中，分析你所观察到的结果。
    epsilons = [0.01, 0.05, 0.1, 0.2]
    
    # 以下绘制epsilon=0.01的曲线图
    iter = 30000
    eps = 0.01
    
    x = np.array(range(iter))
    y = np.zeros(iter, dtype=np.float64)
    
    values = np.zeros(n*2+1, dtype=np.float64)
    counts = np.zeros(n*2+1, dtype=np.int64)
    for i in range(1, iter):
        action = epsilon_greedy(values, eps)
        counts[action] += 1
        value = bandit.pull(action)
        values[action] = (values[action] * (counts[action] - 1) + value) / counts[action]
        y[i] = (y[i-1] * (i-1) + value) / i
    plt.figure()
    plt.plot(x, y)
    plt.xlabel('Iterations')
    plt.ylabel('Average reward')
    plt.show()