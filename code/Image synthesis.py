from astropy.visualization import make_lupton_rgb
import os
import numpy as np

#nomorlization
def noramlization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

path=r'flux data'
 #the data is

files_name=os.listdir(path)
for file_name in files_name:
    file_path=os.path.join(path,file_name)
    data=np.load(file_path)
    data = data[:, 20:54, 20:54,]
    print(data.shape)
    w1=data[0,:,:]
    w1 = noramlization(w1)
    print(w1.shape)
    w2=data[1,:,:]
    w2 = noramlization(w2)
    w3 = data[2, :, :]
    w3 = noramlization(w3)

    name_w=r'E:\红外图像分类\验证数据\涡旋\tuxiang'+'\\'+'wx'+'_'+file_name[:-4]+str(2)+'_'+'.jpg'
    rgb = make_lupton_rgb(w3, w2, w1, Q=2, stretch=0.5, filename=name_w)