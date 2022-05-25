import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
import evaluate as eva
import os
from PIL import Image
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def cal_dis(data, clu, k):
    """
    计算质点与数据点的距离
    :param data: 样本点
    :param clu:  质点集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            dis[i].append(m.sqrt((data[i, 0] - clu[j, 0])**2 + (data[i, 1]-clu[j, 1])**2 +(data[i, 2]-clu[j, 2])**2   ))
    return np.asarray(dis)


def divide(data, dis):
    """
    对数据点分组
    :param data: 样本集合
    :param dis: 质心与所有样本的距离
    :param k: 类别个数
    :return: 分割后样本
    """
    clusterRes = [0] * len(data)
    # print(clusterRes)
    # print(dis.shape)
    for i in range(len(data)):
        seq = np.argsort(dis[i])
        clusterRes[i] = seq[0]
    return np.asarray(clusterRes)


def center(data, clusterRes, k):
    """
    计算质心
    :param group: 分组后样本
    :param k: 类别个数
    :return: 计算得到的质心
    """
    clunew = []
    for i in range(k):
        # 计算每个组的新质心
        idx = np.where(clusterRes == i)
        # print('_____________________')
        # print(data[idx].shape)
        sum = data[idx].sum(axis=0)
        avg_sum = sum/len(data[idx])
        clunew.append(avg_sum)
    clunew = np.asarray(clunew)
    return clunew[:, 0:3]


def classfy(data, clu, k):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param clu: 质心集合
    :param k: 类别个数
    :return: 误差， 新质心
    """
    clulist = cal_dis(data, clu, k)  #计算每一个样本与质心的距离
    clusterRes = divide(data, clulist)  #按距离将样本点进行分割
    clunew = center(data, clusterRes, k)  #重新计算新的质心
    err = clunew - clu
    return err, clunew, k, clusterRes


def plotRes(data, clusterRes, clusterNum):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :param clusterNum: 类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['red', 'blue']#'green', 'yellow', 'red', 'purple', 'orange', 'brown'
    print(clusterRes)
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = [];z1=[]
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
                z1.append(data[j, 2])
        plt.scatter(x1, y1, z1,c=color, alpha=1, marker='+')
        plt.legend()
    plt.show()


if __name__ == '__main__':
    k = 2
    # wx_data = pd.read_excel(r'E:\二分类论文\分类\红外图像\测试集\011.xlsx', header=0)
    # ty_data = pd.read_excel(r'E:\二分类论文\分类\红外图像\测试集\100.xlsx', header=0)
    # data1 = wx_data.values[:, 6:9]
    #
    # data2 = ty_data.values[:, 6:9]


    wx_data = pd.read_excel(r'Magnitude of test dataset\elliptical.xlsx', header=0)
    data1 = wx_data.values[:, 2:5]
    data1_label=wx_data.values[:,8]
    data1 = np.c_[data1, data1_label]
    ty_data = pd.read_excel(r'Magnitude of test dataset\spiral_magnitude.xlsx', header=0)
    data2 = ty_data.values[:, 2:5]
    data2_label=ty_data.values[:,8]
    data2 = np.c_[data2, data2_label]
    data = np.r_[data1, data2]
    print(data[:,3])
    print(data)
    #
    clu = random.sample(data[:, 0:3].tolist(), k)  # 随机取质心
    print(clu)
    clu = np.asarray(clu)
    err, clunew,  k, clusterRes = classfy(data, clu, k)
    while np.any(abs(err) > 0):
        err, clunew,  k, clusterRes = classfy(data, clunew, k)
    clulist = cal_dis(data, clunew, k)
    clusterResult = divide(data, clulist)

    # nmi, acc, purity = eva.eva(clusterResult, np.asarray(data[:, 2]))
    # print(nmi,acc, purity)

    # print('*************************')

    test_epoch_acc = 100 * sum(clusterResult == data[:, 3]) / data[:, 3].shape[0]
    print('test_epoch_acc:', test_epoch_acc)
    cm = confusion_matrix(data[:, 3], clusterResult)
    print('混淆矩阵')
    print(cm)
    target_names = ['tuoyuan','woxuan' ]
    print(classification_report(data[:, 3], clusterResult, target_names=target_names))
    precision = sum(clusterResult == data[:, 3]) / data[:, 3].shape[0]
    print('Test precision: ', precision)
    plotRes(data, clusterResult, k)