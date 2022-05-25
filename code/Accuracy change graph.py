# We save the accuracy curve changes of all models during training and draw them on a picture.

import numpy as np
import matplotlib.pyplot as plt

# WGC network
WGC_trian_acc=np.load(r'train result\WGC\train_acces.npy',allow_pickle=True)
WGC_test_acc=np.load(r'train result\WGC\test_acces.npy',allow_pickle=True)


#no_CBAM
no_CBAM_trian_acc=np.load(r'train result\No_CBAM\train_acces.npy',allow_pickle=True)
no_CBAM_test_acc=np.load(r'train result\No_CBAM\test_acces.npy',allow_pickle=True)

#resnet50
resnet50_trian_acc=np.load(r'train result\resnet50\train_acces.npy',allow_pickle=True)
resnet50_test_acc=np.load(r'train result\resnet50\test_acces.npy',allow_pickle=True)

#vgg19
vgg19_trian_acc=np.load(r'train result\vgg19\train_acces.npy',allow_pickle=True)
vgg19_test_acc=np.load(r'train result\vgg19\test_acces.npy',allow_pickle=True)

x=np.array(range(0,150,1))
# print(zi_CBAM_test_acc)

plt.plot(x,WGC_trian_acc,linestyle='-',color='r',label='WGC_trian_acc')
plt.plot(x,WGC_test_acc,linestyle='--',color='r',label='WGC_val_acc')

plt.plot(x,no_CBAM_trian_acc,linestyle='-',color='g',label='no_CBAM_trian_acc')
plt.plot(x,no_CBAM_test_acc,linestyle='--',color='g',label='no_CBAM_val_acc')

plt.plot(x,resnet50_trian_acc,linestyle='-',color='b',label='resnet50_trian_acc')
plt.plot(x,resnet50_test_acc,linestyle='--',color='b',label='resnet50_val_acc')

plt.plot(x,vgg19_trian_acc,linestyle='-',color='y',label='vgg19_trian_acc')
plt.plot(x,vgg19_test_acc,linestyle='--',color='y',label='vgg19_val_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.savefig('123.jpg')
plt.show()
# trian_loss=np.load(r'E:\论文\结果\自己无CBAM\train_losses.npy',allow_pickle=True)
# trian_acc=np.load(r'E:\论文\结果\自己无CBAM\train_acces.npy',allow_pickle=True)
# test_loss=np.load(r'E:\论文\结果\自己无CBAM\test_losses.npy',allow_pickle=True)
# test_acc=np.load(r'E:\论文\结果\自己无CBAM\test_acces.npy',allow_pickle=True)
# x=np.array(range(0,100,1))
# print(x)
# plt.subplot(121)
# plt.title('Acc')
# plt.plot(x,zi_trian_acc[0:100]/100,label='zi_trian_acc')
# plt.plot(x,zi_test_acc[0:100]/100,label='zi_test_acc')
# plt.plot(x,trian_acc[0:100]/100,label='trian_acc')
# plt.plot(x,test_acc[0:100]/100,label='test_acc')
# plt.legend()
# plt.subplot(122)
# plt.title('loss')
# plt.plot(x,zi_trian_loss[0:100],label='zi_trian_loss')
# plt.plot(x,zi_test_loss[0:100],label='zi_test_loss')
# plt.plot(x,trian_loss[0:100],label='trian_loss')
# plt.plot(x,test_loss[0:100],label='test_loss')
# plt.legend()
# plt.show()


