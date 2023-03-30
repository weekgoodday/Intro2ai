# -*- coding: utf-8 -*-
"""
@ author: Yiliang Liu
"""


# 作业内容：更改loss函数、网络结构、激活函数，完成训练MLP网络识别手写数字MNIST数据集

import numpy as np
import matplotlib.pyplot as plt
from tqdm  import tqdm


# 加载数据集,numpy格式
X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码


# 定义激活函数
def relu(x):
    '''
    relu函数
    '''
    y=np.zeros_like(x)
    y[x>0]=x[x>0]
    return y

def relu_prime(x):
    '''
    relu函数的导数
    '''
    y=np.zeros_like(x)
    y[x>0]=1
    return y

#输出层激活函数
def f(x):
    '''
    softmax函数, 防止除0
    x 按batch输入，每行为一个数据
    利用numpy广播机制： (4,3)除以(4,1) (4,1)广播成(4,3)后对应元素相除
    '''
    return np.exp(x)/np.sum(np.exp(x),axis=1,keepdims=True)

# 定义损失函数
def loss_fn(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    对整个batch求和
    '''
    return -np.sum(y_true*np.log(y_pred))

# 一下两个导数完全可以统一
def f_prime(x):
    '''
    softmax函数的导数
    '''
    pass
def loss_fn_prime(y_true, y_pred):
    '''
    y_true: (batch_size, num_classes), one-hot编码
    y_pred: (batch_size, num_classes), softmax输出
    '''
    pass

def crossentropy_prime(y_true,y_pred):
    '''
    结合了softmax+NLL的总体导数
    '''
    return y_pred-y_true

# 定义权重初始化函数
def init_weights(shape=()):
    '''
    初始化权重
    '''
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

# 定义网络结构
class Network(object):
    '''
    MNIST数据集分类网络
    '''

    def __init__(self, input_size, hidden_size, output_size, lr=0.01,batch_size=64):
        '''
        初始化网络结构
        '''
        self.W1=init_weights(shape=(input_size,hidden_size))
        self.b1=init_weights(shape=(hidden_size,))
        self.W2=init_weights(shape=(hidden_size,output_size))
        self.b2=init_weights(shape=(output_size,))
        self.y1=np.zeros((batch_size,hidden_size))
        self.lr=lr
    def forward(self, x):
        '''
        前向传播
        按batch一起算，肯定比一个个算快
        '''
        batch_size=x.shape[0]
        z1=np.matmul(x,self.W1)+ np.tile(self.b1,reps=(batch_size,1)) # b要广播到(batch_size,hidden_size) 事实证明，不需要显示实现，另一维会自动广播
        y1=relu(z1)
        self.y1=y1
        z2=np.matmul(y1,self.W2)+self.b2
        y2=relu(z2)
        y_pred=f(y2)
        return y_pred


        

    def step(self, x_batch, y_batch):
        '''
        一步训练
        '''
        # 前向传播
        y_pred=self.forward(x_batch)
        # 计算损失和准确率 每个batch
        batch_size=y_pred.shape[0]
        loss=loss_fn(y_batch,y_pred)/batch_size
        acc=np.sum((np.argmax(y_pred,axis=1)==np.argmax(y_batch,axis=1)))/batch_size
        # print("loss:{} batch_acc:{} batch_size:{} lr:{}".format(loss, acc, batch_size, self.lr))
        # 反向传播
        delta_L=crossentropy_prime(y_true=y_batch,y_pred=y_pred)  # (batch,out)
        delta_1=np.matmul(delta_L,self.W2.T)*relu_prime(self.y1) #(batch,hidden)
        # 更新权重
        self.b2=self.b2-self.lr*np.sum(delta_L,axis=0) #(1,out)
        self.W2=self.W2-self.lr*np.matmul(self.y1.T,delta_L) #(hidden,out) 分块矩阵乘法性质，直接乘就是对应相加
        self.b1=self.b1-self.lr*np.sum(delta_1,axis=0) #(1,hidden)
        self.W1=self.W1-self.lr*np.matmul(x_batch.T,delta_1) #(in,hidden)
        return loss,acc,batch_size


if __name__ == '__main__':
    # 训练网络
    net = Network(input_size=784, hidden_size=256, output_size=10, lr=0.01)
    acc_val_list=[]
    acc_test_list=[]
    for epoch in range(10):
        loss_total=0
        acc_total=0
        for i in range (0,len(X_train),64):
            loss_batch,acc_batch,batch_size=net.step(X_train[i:i+64,:],y_train[i:i+64,:])  # 最后一个batch只剩16了
            loss_total=(i*loss_total+batch_size*loss_batch)/(i+batch_size)
            acc_total=(i*acc_total+batch_size*acc_batch)/(i+batch_size)
            if(i%6400==0):
                print("train_epoch:%d, until picture%d, train_loss:%.3f train_acc:%.3f" % (epoch, i, loss_total, acc_total))
        
        # validation 因为要交曲线图直接用测试集了
        acc_val=0
        acc_test=0
        for i in range (0,len(X_val),64):
            y_pred=net.forward(X_val[i:i+64,:])
            batch_size=X_val[i:i+64,:].shape[0]
            acc_batch=np.sum((np.argmax(y_pred,axis=1)==np.argmax(y_val[i:i+64,:],axis=1)))/batch_size
            acc_val=(i*acc_val+batch_size*acc_batch)/(i+batch_size)
        for i in range (0,len(X_test),64):
            y_pred=net.forward(X_test[i:i+64,:])
            batch_size=X_test[i:i+64,:].shape[0]
            acc_batch=np.sum((np.argmax(y_pred,axis=1)==np.argmax(y_test[i:i+64,:],axis=1)))/batch_size
            acc_test=(i*acc_test+batch_size*acc_batch)/(i+batch_size)
        print("epoch:%d, val_acc:%.2f%%"%(epoch,100*acc_val))
        print("epoch:%d, test_acc:%.2f%%"%(epoch,100*acc_test))
        acc_val_list.append(100*acc_val)
        acc_test_list.append(100*acc_test)
    plt.figure(figsize=(10, 5))
    plt.plot(acc_test_list, label='val_acc',alpha=0.5,color='b')
    plt.plot(acc_test_list, label='test_acc',alpha=0.5,color='y')
    plt.plot(np.ones_like(acc_test_list)*94,ls='--',color='grey')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc %")
    plt.title("numpy implement mnist")
    plt.savefig("./numpy_acc.jpg")
        # p_bar = tqdm(range(0, len(X_train), 64))
        # for i in p_bar:
        #     pass
        