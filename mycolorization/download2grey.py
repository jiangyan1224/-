#把下载下来的彩色图片转为灰度图
from glob import glob
import numpy as np

import cv2

data=glob("train/1.jpg")
for imname in data:
    cimg=cv2.imread(imname,1)#以彩色图像导入,3通道 高，宽，（B，G，R）
    print(cimg)
    print("---------------\n")
    print(cimg.reshape(-1,3));
    cimg=np.fliplr(cimg.reshape(-1,3).reshape(cimg.shape))
    cimg=cv2.resize(cimg,(256,256))

    cimg=cv2.imread(imname,0)#以灰度图片读入，1通道