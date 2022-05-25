#Ablation experiment
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



class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            #cnn_layers1
            nn.Conv2d(3,64, kernel_size=6, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # cnn_layers2
            nn.Conv2d(64,128, kernel_size=5, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # cnn_layers3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # cnn_layers4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # cnn_layers5
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.linear_layers = nn.Sequential(
            nn.Linear(256 * 1 * 1, 2048),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2)
        )
        

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), 256*1*1)
        x = self.linear_layers(x)
        return x



model=MyModel()
print(model)

# print the model structure
print(model)
model = model.to('cuda')
loss_func=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
epoch_n =150
train_acces=[]
train_losses=[]
test_acces=[]
test_losses=[]
max_acc =0
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
    if test_epoch_acc > max_acc:
        torch.save(model, r'path' + 'No_CBAM.pt')
        max_acc = test_epoch_acc
train_acces=np.array(train_acces)
train_losses=np.array(train_losses)
test_acces=np.array(test_acces)
test_losses=np.array(test_losses)
np.save(r'train result\No_CBAM\\'+'train_acces'+'.npy',train_acces)
np.save(r'train result\No_CBAM\\'+'train_losses'+'.npy',train_losses)
np.save(r'train result\No_CBAM\\'+'test_acces'+'.npy',test_acces)
np.save(r'train result\No_CBAM\\'+'test_losses'+'.npy',test_losses)