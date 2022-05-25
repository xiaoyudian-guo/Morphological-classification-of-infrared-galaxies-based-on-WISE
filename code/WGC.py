#数据集
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.functional as F
import time
from torchvision import models, transforms
from thop import profile
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from torchvision import transforms, datasets

# train_data_dir = r'E:\红外图像分类\用来红外图像分类的数据\分类\裁剪图像'
# test_data_dir=r''
#
# data_transform=transforms.Compose([transforms.ToTensor()])
# train_datasets=datasets.ImageFolder(root=train_data_dir,transform=data_transform,)
# train_dataloader=torch.utils.data.DataLoader(dataset=train_datasets,batch_size=128,shuffle=True)
#
# test_datasets=datasets.ImageFolder(root=test_data_dir,transform=data_transform)
# test_dataloader=torch.utils.data.DataLoader(dataset=train_datasets,batch_size=128,shuffle=True)

train_data_dir = r'data\enhancement images\train'
test_data_dir = r'data\enhancement images\val'

data_transform = transforms.Compose([transforms.Resize((64,64)),
                                     transforms.RandomHorizontalFlip(),  # 按0.5的概率水平翻转图片,
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_datasets = datasets.ImageFolder(root=train_data_dir, transform=data_transform, )
train_dataloader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=256, shuffle=True)

test_datasets = datasets.ImageFolder(root=test_data_dir, transform=data_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=256, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            # 定义2D卷积层
            nn.Conv2d(3,64, kernel_size=6, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),)
        self.cnn_layers2 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=5, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.linear_layers = nn.Sequential(
            nn.Linear(256*1*1,2048),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(2048,2)
        )
        self.CBAM1=CBAM(64)
        self.CBAM2 = CBAM(256)
        # 前项传播

    def forward(self, x):
        x = self.cnn_layers(x)
        x= self.CBAM1(x)
        x = self.cnn_layers2(x)
        x = self.CBAM2(x)
        x = x.view(x.size(0), 256*1*1)
        x = self.linear_layers(x)
        return x



model=MyModel()
print(model)

# 打印模型，呈现网络结构
print(model)
model = model.to('cuda')
loss_func=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epoch_n =150
train_acces=[]
train_losses=[]
test_acces=[]
test_losses=[]
max_acc=0
for epoch in range(epoch_n):
    print('epoch {}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)
    model.train(True)
    train_loss = 0.0
    train_corrects = 0.0
    for batch, data in enumerate(train_dataloader, 1):
        X, Y = data
        X, Y = X.cuda(), Y.cuda()
        y_pred = model(X)
        _, pred = torch.max(y_pred.data, 1)
        optimizer.zero_grad()
        loss = loss_func(y_pred, Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.data.item()
        train_corrects += torch.sum(pred == Y.data)
    train_epoch_loss = train_loss  / len(train_dataloader)
    train_losses.append(train_epoch_loss)
    train_epoch_acc = 100 * train_corrects / len(train_datasets)
    train_acces.append(train_epoch_acc)
    print('train: Loss:{:.4f} Acc:{:.4f}%'.format(train_epoch_loss, train_epoch_acc))

    model.eval()
    test_loss = 0.0
    test_corrects = 0.0
    for batch, data in enumerate(test_dataloader, 1):
        X, Y = data
        X, Y = X.cuda(), Y.cuda()
        y_pred = model(X)
        _, pred = torch.max(y_pred.data, 1)
        optimizer.zero_grad()
        loss = loss_func(y_pred, Y)
        test_loss += loss.data.item()
        test_corrects += torch.sum(pred == Y.data)
    test_epoch_loss = test_loss / len(test_dataloader)
    test_losses.append(test_epoch_loss)
    test_epoch_acc = 100 * test_corrects / len(test_datasets)
    test_acces.append(test_epoch_acc )
    print('test: Loss:{:.4f} Acc:{:.4f}%'.format(test_epoch_loss, test_epoch_acc))
    if test_epoch_acc >max_acc:
        torch.save(model,r'train result\WGC\\'+ 'CBAM.pt')
        max_acc = test_epoch_acc
train_acces=np.array(train_acces)
train_losses=np.array(train_losses)
test_acces=np.array(test_acces)
test_losses=np.array(test_losses)
np.save(r'train result\WGC\\'+'train_acces'+'.npy',train_acces)
np.save(r'train result\WGC\\'+'train_losses'+'.npy',train_losses)
np.save(r'train result\WGC\\'+'test_acces'+'.npy',test_acces)
np.save(r'train result\WGC\\'+'test_losses'+'.npy',test_losses)

