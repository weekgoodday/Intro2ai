# 用Pytorch手写一个LSTM网络，在IMDB数据集上进行训练

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from utils import load_imdb_dataset, Accuracy
import sys

from utils import load_imdb_dataset, Accuracy

use_mlu = False
try:
    import torch_mlu
    import torch_mlu.core.mlu_model as ct
    global ct
    use_mlu = torch.mlu.is_available()
except:
    use_mlu = False

if use_mlu:
    device = torch.device('mlu:0')
else:
    print("MLU is not available, use GPU/CPU instead.")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')



X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)

seq_Len = 200
vocab_size = len(X_train) + 1
n_epoch = 5
batch_size = 32
print_freq = 2
word_embedding_dim = 64
hidden_size = 64 

class ImdbDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):

        data = self.X[index]
        data = np.concatenate([data[:seq_Len], [0] * (seq_Len - len(data))]).astype('int32')  # set
        label = self.y[index]
        return data, label

    def __len__(self):

        return len(self.y)

class LSTM(nn.Module):
    '''
    手写lstm，可以用全连接层nn.Linear，不能直接用nn.LSTM
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # LSTM层
        self.input_size = input_size
        self.W_i = nn.Linear(input_size+hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size+hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size+hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size+hidden_size, hidden_size)

    def forward(self, x, init_states): # (betch, sequence, feature)
        if(init_states is None):
            h_t,c_t = self._init_states(x)
        else:
            h_t,c_t = init_states
        batch_size, seq_size, _ = x.size()
        hidden_seq = []
        for t in range(seq_size):
            combined = torch.cat((h_t, x[:, t, :]), dim=1) #h_t (batch, hidden_size) x[:, t, :] (batch, input_size) combined (32,128)
            i_t = torch.sigmoid(self.W_i(combined))
            f_t = torch.sigmoid(self.W_f(combined))
            o_t = torch.sigmoid(self.W_o(combined))
            g_t = torch.tanh(self.W_c(combined))
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t) # (32, 64)
            hidden_seq.append(h_t.unsqueeze(0)) # list 每个元素 [1, 32, 64]
        hidden_seq = torch.cat(hidden_seq, dim=0) # list 合并为tensor [200, 32, 64]
        hidden_seq = hidden_seq.transpose(0, 1).contiguous() # 变为 [32, 200, 64]
        return hidden_seq, (h_t, c_t)
    
    def _init_states(self, x):
        h_t = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype).to(x.device) # x.size(0)是batch_size
        c_t = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype).to(x.device)
        return h_t,c_t

out_embedding_dim=64 # 也即hidden_size
mlp_embedding_dim=64
class Net(nn.Module):
    '''
    一层LSTM的文本分类模型
    '''
    def __init__(self, embedding_size=64, hidden_size=64, num_classes=2):
        super(Net, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # LSTM层
        self.lstm = LSTM(input_size=hidden_size, hidden_size=hidden_size)
        # 全连接层
        # self.fc1 = nn.Linear(hidden_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, num_classes)
        self.linear1 = nn.Linear(in_features=out_embedding_dim, out_features=mlp_embedding_dim)
        self.act1 = torch.nn.ReLU()
        self.linear2 = nn.Linear(in_features=mlp_embedding_dim, out_features=2)
    def forward(self, x):
        '''
        x: 输入, shape: (seq_len, batch_size)
        '''

        # 词嵌入
        x = self.embedding(x)
        # LSTM层
        x, _ = self.lstm(x, None)
        x = torch.mean(x, dim=1) # [32, 200, 64] --> [32, 64]
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x





train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

net = Net()
metric = Accuracy()
print(net)

def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter= 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.long()
        data, target = data.to(device), target.to(device)  # data [64,200] target [64]
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output,target)
        train_acc += metric.result()
        train_loss +=  loss.item()
        metric.reset()
        n_iter+=1
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(
        epoch, train_loss / n_iter,train_acc/n_iter))
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)
