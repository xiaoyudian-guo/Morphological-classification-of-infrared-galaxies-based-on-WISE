import os
from torchvision import models, transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
data_dir = r'E:\二分类论文\分类\增强图像\验证集\0\wx_19.63438683_-1.197360869_.jpg'

data_transform = transforms.Compose([transforms.Resize((64,64)),
                                     # transforms.RandomHorizontalFlip(),  # 按0.5的概率水平翻转图片,
                                     # transforms.ToTensor(),
                                     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
data_transform1 = transforms.Compose([#transforms.Resize((64,64)),
                                     transforms.RandomHorizontalFlip(),  # 按0.5的概率水平翻转图片,
                                     # transforms.ToTensor(),
                                     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])
data_transform2 = transforms.Compose([#transforms.Resize((64,64)),
                                     #transforms.RandomHorizontalFlip(),  # 按0.5的概率水平翻转图片,
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])

img=Image.open(data_dir)
img1=data_transform(img)
img2=data_transform1(img1)
img3=data_transform2(img2)
plt.subplot(131)
plt.imshow(img)
plt.title('Original image')
plt.subplot(132)
plt.imshow(img1)
plt.title('Upsample')
plt.subplot(133)
plt.imshow(img2)
plt.title('Random rotation')
# plt.subplot(224)
# plt.imshow(img3)
# plt.title('归一化')


plt.show()
