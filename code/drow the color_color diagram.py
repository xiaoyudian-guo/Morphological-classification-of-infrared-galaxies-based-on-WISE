#drow the color-color diagram of test



import numpy as np
from cv2 import cv2
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn import metrics
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns
spiral_data = pd.read_excel(r'Magnitude of test dataset\spiral_magnitude.xlsx', header=0)
elliptical_data = pd.read_excel(r'Magnitude of test dataset\elliptical.xlsx', header=0)

# wx_data = pd.read_csv(r'E:\二分类论文\分类\红外图像\测试集\0.csv', header=0)
data1_w1_w2 = spiral_data.values[:,6]
data1_w2_w3 = spiral_data.values[:,7]
# ty_data = pd.read_csv(r'E:\二分类论文\分类\红外图像\测试集\1.csv', header=0)
data2_w1_w2 = elliptical_data.values[:,6]
data2_w2_w3 = elliptical_data.values[:,7]


p1 = sns.kdeplot(data1_w2_w3, data1_w1_w2, cmap="Reds", shade=True, bw=.15)
p2 = sns.kdeplot(data2_w2_w3, data2_w1_w2, cmap="Blues", shade=True, bw=.15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('W2-W3',fontsize=13)
plt.ylabel('W1-W2',fontsize=13)
plt.legend(labels=['Elliptical','Spiral'])
plt.show()

# data1=wx_data.values[:,2:6]
# data2=ty_data.values[:,2:6]
# print(data1.shape)
# plt.figure(figsize=(8,6))  # 设置画布大小
# ax = plt.axes(projection='3d')  # 设置三维轴
# ax.scatter(data2[:,1],data2[:,2],data2[:,3],c='b',alpha=0.5)
# ax.scatter(data1[:,1],data1[:,2],data1[:,3],c='r',alpha=0.5)
#
# plt.show()


# def plot_decision_boundary(svm,x_text,y_text):
#     '''
#     函数功能，将svm训练出的结果可视化函数，参数说明，
#     svm：训练好的svm模型
#     x_text：测试数据集
#     y_text：测试数据集标签
#     '''
#     #  确定x轴上的左右边界
#     x_min,x_max = x_text[:,0].min()-1,x_text[:,0].max()+1
#     #  确定y轴上的左右边界
#     y_min,y_max = x_text[:,1].min()-1,x_text[:,1].max()+1
#     #  创建合适的网格，h是步长
#     h =0.02
#     xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
#                         np.arange(y_min,y_max,h))
#     #  np.c_功能，按行连接两个矩阵，要求行数一样。
#     x_hypo = np.c_[xx.ravel().astype(np.float32),
#                    yy.ravel().astype(np.float32)]
#     #  开始预测
#     _,zz = svm.predict(x_hypo)
#     #  大小要一样
#     zz = zz.reshape(xx.shape)
#     #  绘制三维等高线图，必须在网格结构中才可以
#     plt.contourf(xx,yy,zz,cmap=plt.cm.coolwarm,alpha=0.8)
#     #  按标签绘制散点图
#     plt.scatter(x_text[:,0],x_text[:,1],c=y_text,s=100)
#
#
#
#
# if __name__ == '__main__':
#     wx_data = pd.read_csv(r'E:\二分类论文\分类\红外图像\测试集\0.csv', header=0)
#     data1 = wx_data.values[:, 6:9]
#
#     ty_data = pd.read_csv(r'E:\二分类论文\分类\红外图像\测试集\1.csv', header=0)
#     data2 = ty_data.values[:,6:9]
#     data = np.r_[data1,data2]
#     np.random.shuffle(data)
#     x=data[:,0:2]
#     y=data[:,2]
#     #  使用自带函数创建一个含有100个样本，特征两个，两个标签的数据样本
#     # x,y = datasets.make_classification(n_samples=100,n_features=2,n_classes=2,n_redundant=0,random_state=7816)
#     print(x)
#     print(y)
#     x = x.astype(np.float32)
#     # y = y*2-1
#     # print(y)
# #  定义一个含各种svm核的列表
#     kernels = [cv2.ml.SVM_LINEAR,cv2.ml.SVM_INTER,
#            cv2.ml.SVM_SIGMOID,cv2.ml.SVM_RBF]
#     #  分割数据集，20%为测试集
#     x_train,x_text,y_train,y_text = ms.train_test_split(x,y,test_size=0.2,random_state=42)
#     #  训练含有不同核的svm分类器
#     for i,kernel in enumerate(kernels):
#         svm = cv2.ml.SVM_create()
#         #  设置svm核
#         svm.setKernel(kernel)
#         svm.train(x_train,cv2.ml.ROW_SAMPLE,y_train)
#         b,y_pred = svm.predict(x_text)
#         #  用测试集计算准确率
#         a=metrics.accuracy_score(y_text,y_pred)
#         print(a)
#         #  创建4个子图
#         plt.subplot(2,2,i+1)
#         #  可视化结果
#         plot_decision_boundary(svm,x_text,y_text)
#         plt.title('accuracy: %.2f' %a)
#     plt.show()