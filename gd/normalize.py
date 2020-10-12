import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
filepath = '../iemocap/spectrogram'  # 数据集目录
pathDir=[]
R_channel = 0
G_channel = 0
B_channel = 0
imagenum=0
sessionlist = os.listdir(filepath)
for session in sessionlist:
    sessionpath=os.path.join(filepath,session)
    filelist=os.listdir(sessionpath)
    for file in filelist:
        imagepath=os.path.join(sessionpath,file)
        pathDir=os.listdir(imagepath)
        for idx in range(len(pathDir)):
            filename = pathDir[idx]
            imagenum+=1
            img = imread(os.path.join(imagepath, filename))
            R_channel = R_channel + np.sum(img[:, :, 0])
            G_channel = G_channel + np.sum(img[:, :, 1])
            B_channel = B_channel + np.sum(img[:, :, 2])

num = imagenum * 496 * 369  # 这里（1024,1024）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num  # or /255.0
G_mean = G_channel / num
B_mean = B_channel / num

R_channel = 0
G_channel = 0
B_channel = 0
for session in sessionlist:
    sessionpath=os.path.join(filepath,session)
    filelist=os.listdir(sessionpath)
    for file in filelist:
        imagepath=os.path.join(sessionpath,file)
        pathDir=os.listdir(imagepath)
        for idx in range(len(pathDir)):
            filename = pathDir[idx]
            img = imread(os.path.join(imagepath, filename))
            R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
            G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
            B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / num)
G_var = np.sqrt(G_channel / num)
B_var = np.sqrt(B_channel / num)
print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))

# R_mean is 143.532975, G_mean is 145.831770, B_mean is 151.186388
# R_var is 48.226279, G_var is 45.276815, B_var is 40.371132
