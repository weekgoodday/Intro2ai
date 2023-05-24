# The same set of code can switch the backend with one line
import os

import torch
from torch.nn import Module
from torch.nn import Linear, LSTM, Embedding
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import sys
from utils import load_imdb_dataset, Accuracy
import os
print(os.getcwd())
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
n_epoch = 5
batch_size = 32
print_freq = 2
seq_Len = 200 #200截断与补0
word_embedding_dim=64
lstm_layers=1
bidirectional_lstm=False
if bidirectional_lstm:
    D=2
else:
    D=1
prev_h = np.random.random([D*lstm_layers, batch_size, word_embedding_dim]).astype(np.float32)
prev_c = np.random.random([D*lstm_layers, batch_size, word_embedding_dim]).astype(np.float32)
prev_h = torch.FloatTensor(prev_h).to(device)
prev_c = torch.FloatTensor(prev_c).to(device)

X_train, y_train, X_test, y_test = load_imdb_dataset('data', nb_words=20000, test_split=0.2)
# X and y are all 20000 length;  X is a list of list, y is a list of 0/1
# X[0] length 140 X[1] length 268 
# test X 5000 length

vocab_size = len(X_train) + 1
print("vocab_size: ", vocab_size)


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

out_embedding_dim=64
mlp_embedding_dim=64
class ImdbNet(Module):

    def __init__(self):
        super(ImdbNet, self).__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)
        self.lstm = LSTM(input_size=word_embedding_dim, hidden_size=out_embedding_dim, batch_first=True,bidirectional=bidirectional_lstm)
        self.linear1 = Linear(in_features=out_embedding_dim, out_features=mlp_embedding_dim)
        self.act1 = torch.nn.ReLU()
        self.linear2 = Linear(in_features=mlp_embedding_dim, out_features=2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x, [prev_h, prev_c]) #[32,200,64]
        x = torch.mean(x, dim=1) #[32,64] 取平均 训练loss的时候也是mean
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x




train_dataset = ImdbDataset(X=X_train, y=y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

net = ImdbNet()
metric = Accuracy()
print(net)


def train(model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    train_acc = 0
    train_loss = 0
    n_iter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.long()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        metric.update(output, target)
        train_acc += metric.result()
        train_loss += loss.item()
        metric.reset()
        n_iter += 1
    print('Train Epoch: {} Loss: {:.6f} \t Acc: {:.6f}'.format(epoch, train_loss / n_iter, train_acc / n_iter))


optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0)
gamma = 0.7
for epoch in range(1, n_epoch + 1):
    train(net, device, train_loader, optimizer, epoch)