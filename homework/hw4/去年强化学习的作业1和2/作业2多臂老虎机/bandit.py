import numpy as np
from random import random, randint
from time import sleep
from matplotlib import pyplot as plt
#多臂老虎机 k-armed bandits
class NormalDistBandit:
    def __init__(self, means, stds):
        assert len(means) == len(stds), "Means and stds must be the same length."
        self.n = len(means) #n取11
        self.means = np.array(means)
        self.stds = np.array(stds)
        assert all(self.stds >= 0), "Stds must be positive."
    
    def pull(self, k): #相当于拉杆
        assert 0 <= k < self.n, f"Invalid arm {k}."
        return np.random.normal(loc=self.means[k], scale=self.stds[k])

def epsilon_greedy(values, epsilon):
    assert len(values) > 1, "There should be 2 or more values."
    eps = epsilon * len(values) / (len(values) - 1) #减去后面randint rand出最大value的影响
    if random() <= eps: #random均匀0-1 有eps概率探索新的
        return randint(0, len(values)-1) #有1/11探索回max 已经扣除
    return int(np.argmax(values))
   
def UCB(values,counts,t,c):
    ucbvalues=np.array(values)
    for i in range(len(values)):
        if(counts[i]==0):
            return i
    for i in range(len(values)):
        ucbvalues[i]=values[i]+c*(np.log(t)/counts[i])**0.5
    return int(np.argmax(ucbvalues))
    
def random_pick(items,probabilities):
    x=random()
    cummulative_probability=0
    for item,pro in zip(items,probabilities):
        cummulative_probability+=pro
        if(cummulative_probability>x):
            break
    return item



if __name__ == "__main__":
    n = 5
    #共11个 均值-5 -4 -3 -2 -1 0 1 2 3 4 5 方差1 1 1 1 1 1 1 1 1 1 1
    bandit = NormalDistBandit(means = np.array(range(-n, n+1)), stds = np.ones(11))
    
    # Task：把下面这几种epsilon的曲线画到一张图中，分析你所观察到的结果。
    epsilons = [0.01, 0.05, 0.1, 0.2]
    
    # 以下绘制epsilon=0.01的曲线图
    iter = 10000
    #eps = 0.01
    fig,ax = plt.subplots()
    for eps in epsilons:
            
        x = np.array(range(iter)) #这是为了画图 横坐标
        y = np.zeros(iter, dtype=np.float64) #画图的纵坐标average reward
        
        values = np.zeros(n*2+1, dtype=np.float64) #即Q the estimation of bandit
        counts = np.zeros(n*2+1, dtype=np.int64) #记录拉杆次数以便增量更新
        for i in range(1, iter):
            action = epsilon_greedy(values, eps)
            counts[action] += 1
            value = bandit.pull(action) #真实reward Rt(action)
            values[action] = (values[action] * (counts[action] - 1) + value) / counts[action] #更新公式==((n-1)*Qn+Rn)/n==Qn+1/n*(Rn-Qn)
            y[i] = (y[i-1] * (i-1) + value) / i #记录average reward 也是增量法记录，只和reward有关
        ax.plot(x, y,label="epsilon:"+str(eps))
    plt.xlabel('Iterations')
    plt.ylabel('Average reward')
    plt.legend()
    plt.show()

    fig,ax = plt.subplots(num=2) #subplots就会新开一个
#e-gradient e取0.05
    x = np.array(range(iter)) #这是为了画图 横坐标
    y1 = np.zeros(iter, dtype=np.float64) #画图的纵坐标average reward
    values=np.zeros(n*2+1,dtype=np.float64)
    counts=np.zeros(n*2+1,dtype=np.int64) 
    for i in range(1, iter):
        action = epsilon_greedy(values, 0.05) 
        counts[action] += 1
        value = bandit.pull(action) #真实reward Rt(action)
        values[action] = (values[action] * (counts[action] - 1) + value) / counts[action] #更新公式==((n-1)*Qn+Rn)/n==Qn+1/n*(Rn-Qn)
        y1[i] = (y1[i-1] * (i-1) + value) / i #记录average reward 也是增量法记录，只和reward有关
    ax.plot(x, y1,label="e-gradient, e=0.05")
#优化初值
    x = np.array(range(iter)) #这是为了画图 横坐标
    y2 = np.zeros(iter, dtype=np.float64) #画图的纵坐标average reward
    values=np.zeros(n*2+1,dtype=np.float64)+5 #Q初值估计优化初值
    counts=np.zeros(n*2+1,dtype=np.int64) 
    for i in range(1, iter):
        action = epsilon_greedy(values, 0) #优化初值不需要探索
        counts[action] += 1
        value = bandit.pull(action) #真实reward Rt(action)
        values[action] = (values[action] * (counts[action] - 1) + value) / counts[action] #更新公式==((n-1)*Qn+Rn)/n==Qn+1/n*(Rn-Qn)
        y2[i] = (y2[i-1] * (i-1) + value) / i #记录average reward 也是增量法记录，只和reward有关
    ax.plot(x, y2,label="Optimistic Initial Values, initial=5")
#Upper-Confidence-Bound Action Selection
    c=2 #超参数
    x = np.array(range(iter)) #这是为了画图 横坐标
    y3 = np.zeros(iter, dtype=np.float64) #画图的纵坐标average reward
    values=np.zeros(n*2+1,dtype=np.float64)
    counts=np.zeros(n*2+1,dtype=np.int64) 
    for i in range(1,iter):
        action=UCB(values,counts,i,c)
        counts[action]+=1
        value=bandit.pull(action)
        values[action]=(values[action]*(counts[action]-1)+value)/counts[action]
        y3[i]=(y3[i-1]*(i-1)+value)/i
    ax.plot(x,y3,label="UCB, c=2")

#Gradient Bandit Algorithms
    alpha=0.1 #超参数 H更新步长
    x=np.array(range(iter))
    y4=np.zeros(iter,dtype=np.float64)
    h_prefer=np.zeros(n*2+1,dtype=np.float64)
    pi_prefer=np.zeros(n*2+1,dtype=np.float64)+1/n
    for i in range(1,iter):
        action=random_pick(np.array(range(2*n+1)),pi_prefer)
        value=bandit.pull(action) #Rt
        y4[i]=(y4[i-1]*(i-1)+value)/i #average_Rt
        h_prefer[action]+=alpha*(value-y4[i])*(1-pi_prefer[action])
        for a in range(2*n+1):
            if(a!=action):
                h_prefer[a]-=alpha*(value-y4[i])*pi_prefer[a]
        pi_prefer=np.exp(h_prefer)/np.sum(np.exp(h_prefer))
    ax.plot(x,y4,label="Gradient Bandit Algorithm, alpha=0.1")
    plt.xlabel('Iterations')
    plt.ylabel('Average reward')
    plt.legend()
    plt.show()


'''
观察到的现象：
1、观察某一次实验最终10000代的平均reward，epsilon取0.05最高，取0.1其次，取0.01再次，取0.2最低。
但多次实验发现epsilon=0.01也有可能最终average reward最高。
2、就收敛速度来看epsilon取0.2和0.1最快，0.05其次，0.01到最后也没能完全收敛
3、eps取0.01的曲线最为平滑，其它曲线随eps增加变得陡峭
4、所有的情况前几代reward都几乎呈直线迅速提高，甚至epsilon取0.2时，在前几代有明显的冲激。
原因分析：
1、epsilon越小，最后的探索成本越低，在确定了期望估计值后，也就能带来更大的平均reward。也
就是说，当各老虎机的期望估值基本稳定后，iterations取得无限大的时候，理论上epsilon=0.01的
average reward一定会>0.05>0.1>0.2。但是，因为iteration最终取在了10000代，对单次实验来说，
eps=0.1的average reward可能还没收敛，受到探索速度慢，前4000代reward低的影响，最终排在了
eps=0.1与eps=0.2之间。
2、因为epsilon在合理范围内，其取值越大，每代选择探索的可能性越大，收敛速度也就越快。
3、与2的理由类似，eps取值小，探索概率低，reward倾向于维持现已探索到的最大reward eps=0.01相邻
几代reward改变小，average reward的变化慢，因此平滑。
4、因为前几代是探索新老虎机的高峰，这时总的reward基数也不高，一旦探索到reward大的新机，average reward
会迅速受到影响跟进，尤其eps=0.2的时候，此时受探索成本的影响还不稳定，可能会有尖峰。
'''
'''
bonus 观察到的现象与分析：
在已有超参数下，Optimistic Initial Value、UCB、Gradient Bandit的最终average reward都要好于e-gradient的方法，
原因是最终基本根据已有尝试确定estimation之后，三个方法探索的比例低于e-gradient固定的0.05，最后稳定在已知最优的拉杆上。
最后总结一下另外三种方法：
优化初值很好理解，通过提高其它每个拉杆的期望，以加快最开始的，让决策迅速收敛到最好的拉杆并维持。
UCB通过附加项描述uncertainty，探索足够多次，uncertainty项就会很小，从而不会探索。
gradient bandit通过另一种根据reward和平均的差更新选择概率，并最终收敛，数学上能证明类似随机梯度下降。
'''