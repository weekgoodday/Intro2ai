# 第二课作业
# 用pytorch实现卷积神经网络，对cifar10数据集进行分类
# 要求:1. 使用pytorch的nn.Module和Conv2d等相关的API实现卷积神经网络
#      2. 使用pytorch的DataLoader和Dataset等相关的API实现数据集的加载
#      3. 修改网络结构和参数，观察训练效果
#      4. 使用数据增强，提高模型的泛化能力

import os
import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

from cifar_loader import CIFAR10  # 魔改pytorch源码的cifar10数据集加载器 请把魔改过的cifar_loader.py放在当前目录下
import numpy as np  # 只是记录acc用了个append
import matplotlib.pyplot as plt  # 只是画了个训练过程中的acc曲线

# 定义超参数
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

# 定义数据预处理方式
# 数据预处理
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),  # 归一化
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomErasing(scale=(0.04,0.2), ratio=(0.5,2)),  # 随机遮挡
    transforms.RandomCrop(32,padding=4),  # 随机裁剪
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# 定义数据集
train_dataset = CIFAR10(root='./data', train='train', download=True, transform=train_transform)
valid_dataset = CIFAR10(root='./data', train='valid', download=True, transform=test_transform)  # 取1/5的训练数据用作验证集，当然这样的效果肯定比不过全集训练
test_dataset = CIFAR10(root='./data', train='test', download=True, transform=test_transform)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class Net(nn.Module):
    '''
    定义卷积神经网络,3个卷积层,2个全连接层
    '''
    def __init__(self,):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512*2*2,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x) #16
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) #8
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x) #4
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x) #2
        x = self.classifier(x)
        return x



# 实例化模型
model = Net()

use_mlu = False
try:
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

model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 70, 90], gamma=0.8)  # 事实证明这个scheduler对结果没啥本质影响
save_root='./model/'
best_acc = 0
best_epoch = 0
acc_record=np.empty(0)
# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    for i, (images, labels) in enumerate(train_loader):  # 只取前 40000 images训练
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (outputs.argmax(1) == labels).float().mean()

        # 打印训练信息
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), accuracy.item() * 100))
    scheduler.step()

    # 在验证集上测试模型
    model.eval()
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:  # 10000 images
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 valid images: {} %'.format(100 * correct / total))
        acc = correct / total
        acc_record=np.append(acc_record,acc)
        if acc > best_acc:
            best_acc=acc
            best_epoch=epoch+1
            print("epoch:%d,best_acc:%.2f%%" % (epoch+1,best_acc*100))
            
            save_dict={
                'epoch':epoch+1,
                'optimizer_state_dict':optimizer.state_dict(),
            }
            save_dict['model_state_dict']=model.state_dict()
            torch.save(save_dict, save_root+'best_model.ckpt')
np.save(save_root+'acc.npy',acc_record)
fig=plt.figure()
plt.plot(acc_record)
fig.savefig('acc.png')

print("100 epochs training, best_epoch:%d, best_valid_acc:%.2f%%" % (best_epoch,best_acc*100))

# 取验证集上最好的用于测试
state_dict=torch.load(save_root+'best_model.ckpt')
model.load_state_dict(state_dict['model_state_dict'])
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:  # 10000 images
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))