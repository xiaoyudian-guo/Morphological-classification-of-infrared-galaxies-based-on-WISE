
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import time
from torchvision import models, transforms, datasets

resnet50=models.resnet50()
print(resnet50)

train_data_dir = r'data\enhancement images\train'
test_data_dir = r'data\enhancement images\val'
data_transform = transforms.Compose([transforms.Resize((64,64)),
                                     # transforms.Resize(50),
                                     transforms.RandomHorizontalFlip(),  # 按0.5的概率水平翻转图片
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_datasets = datasets.ImageFolder(root=train_data_dir, transform=data_transform, )
train_dataloader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=256, shuffle=True)

test_datasets = datasets.ImageFolder(root=test_data_dir, transform=data_transform)
test_dataloader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=256, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Models(torch.nn.Module):
    def __init__(self,net):
        super(Models, self).__init__()
        self.net=net
        self.Conv =torch.nn.Sequential(
            torch.nn.Linear(1000,2),
        )
    def forward(self, inputs):
        x=self.net(inputs)
        x = self.Conv(x)
        return x
model=Models(resnet50)
model = model.to('cuda')
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
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
    test_epoch_loss = test_loss  / len(test_dataloader)
    test_losses.append(test_epoch_loss)
    test_epoch_acc = 100 * test_corrects / len(test_datasets)
    test_acces.append(test_epoch_acc )
    print('test: Loss:{:.4f} Acc:{:.4f}%'.format(test_epoch_loss, test_epoch_acc))
    if test_epoch_acc > max_acc:
        torch.save(model, r'train result\resnet50' + 'resnet50.pt')   #change the path
        max_acc = test_epoch_acc

train_acces=np.array(train_acces)
train_losses=np.array(train_losses)
test_acces=np.array(test_acces)
test_losses=np.array(test_losses)
np.save(r'train result\resnet50'+'train_acces'+'.npy',train_acces)
np.save(r'train result\resnet50'+'train_losses'+'.npy',train_losses)
np.save(r'train result\resnet50'+'test_acces'+'.npy',test_acces)
np.save(r'train result\resnet50'+'test_losses'+'.npy',test_losses)
