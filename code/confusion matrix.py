#the confusion matrix of different models. you can change the model to get different confusion matrix of model

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
import os
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torchvision import models, transforms, datasets
from thop import profile

# class ChannelAttentionModule(nn.Module):
#     def __init__(self, channel, ratio=16):
#         super(ChannelAttentionModule, self).__init__()
#         # 使用自适应池化缩减map的大小，保持通道不变
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(channel, channel // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // ratio, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x))
#         maxout = self.shared_MLP(self.max_pool(x))
#         return self.sigmoid(avgout + maxout)
#
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # map尺寸不变，缩减通道
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
#
#
# class CBAM(nn.Module):
#     def __init__(self, channel):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(channel)
#         self.spatial_attention = SpatialAttentionModule()
#
#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out
#
#
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.cnn_layers = nn.Sequential(
#             # 定义2D卷积层
#             nn.Conv2d(3,64, kernel_size=6, stride=1, padding=1,bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),)
#         self.cnn_layers2 = nn.Sequential(
#             nn.Conv2d(64,128, kernel_size=5, stride=1, padding=1,bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#         )
#         self.linear_layers = nn.Sequential(
#             nn.Linear(256*1*1,2048),
#             nn.Dropout(),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, 2048),
#             nn.Dropout(),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048,2)
#         )
#         self.CBAM1=CBAM(64)
#         self.CBAM2 = CBAM(256)
#         # 前项传播
#
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x= self.CBAM1(x)
#         x = self.cnn_layers2(x)
#         x = self.CBAM2(x)
#         x = x.view(x.size(0), 256*1*1)
#         x = self.linear_layers(x)
#         return x



# when calculate the confusion matrix of resnet or vgg19,using the following code


class Models(torch.nn.Module):
    def __init__(self,net):
        super(Models, self).__init__()
        self.net=net
        self.Conv =torch.nn.Sequential(
            # torch.nn.Flatten(),

            torch.nn.Linear(1000,2),
        )
    def forward(self, inputs):
        x=self.net(inputs)
        x = self.Conv(x)
        # x=F.
        return x
resnet=torch.load(r'E:\二分类论文\结果\vgg19\\vgg19.pt')

resnet.eval()
print(resnet)
resnet=resnet.cpu()
test_data_dir = r'E:\二分类论文\分类\增强图像\测试集'


input = torch.randn(1, 3, 64, 64)
flops, params = profile(resnet, inputs=(input,))
print(flops)
print(params)

# data_transform = transforms.Compose([transforms.Resize((64,64)),
#                                      transforms.RandomHorizontalFlip(),  # 按0.5的概率水平翻转图片,
#                                      transforms.ToTensor(),
#                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#
# files = os.listdir(test_data_dir)
# def load_pic():
#     label=[]
#     image=[]
#     for file in files:
#         print(type(file))
#         file_path = os.listdir(test_data_dir + '/' + file)
#         for datas in file_path:
#             data_path = test_data_dir + '/' + file + '/' + datas
#             img= Image.open(data_path)
#             img=data_transform(img)       #3,64,64
#             img=img.numpy()
#
#             image.append(img)
#             label.append(file)
#     label = list(map(int, label))
#     image = np.array(image)
#     label = np.array(label)
#     return image,label
# data,label=load_pic()
#
# print(label.shape)
# data= torch.from_numpy(data)
# data = data.float()
# out= resnet(data)
# prediction = torch.max(out, 1)[1]
# prediction=prediction.numpy()
# print(prediction)
# print(prediction.shape)
# test_epoch_acc = 100 * sum(prediction== label)/ label.shape[0]
# print('test_epoch_acc:', test_epoch_acc)
# cm = confusion_matrix(label , prediction )
# print('混淆矩阵')
# print(cm)
# target_names = ['woxuan','tuoyuan']
# print(classification_report(label , prediction , target_names=target_names))
# precision = sum(prediction== label) / label.shape[0]
# print('Test precision: ', precision)






