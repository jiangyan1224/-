# #把下载下来的彩色图片转为灰度图

import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from random import randint

data = glob("train/*.jpg")
for imname in data:

    cimg = cv2.imread(imname,1)#以彩色模式读入图像，3通道 高，宽，（B，G，R）
    # print(cimg.shape)#512,512,3

    # 如果不带这一行，所有图片都偏蓝色
    #CV2的imread默认存储的颜色空间顺序是BGR，imshow的颜色顺序RGB，所以需要把R B互换位置，输出才不会偏蓝色
    cimg = np.fliplr(cimg.reshape(-1,3)).reshape(cimg.shape)
    # print(cimg.shape)#512,512,3
    cimg = cv2.resize(cimg, (256,256))
    # print(cimg.shape)#256,256,3   跟原图的区别只有变糊了


    img = cv2.imread(imname,0)#以灰度模式读入图像
    # print(img.shape)#512,512

    # kernel = np.ones((5,5),np.float32)/25
    # for i in range(30):
    #     randx = randint(0,205)#生成0-205间的随机整数
    #     randy = randint(0,205)
    #     cimg[randx:randx+50, randy:randy+50] = 255#彩色图片中随机一些方块置为白色
    blur = cv2.blur(cimg,(100,100))#均值滤波 减弱图片噪音；100，100是指均值滤波的方框大小


    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #自适应阈值二值化param:原图(一般为灰度图)，当超过阈值所取得值，阈值的计算方法，二值化操作的类型，图片中分块的大小，计算方法的常数项
    img_edge = cv2.adaptiveThreshold(img, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    blockSize=9,
                                    C=2)
    # img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    # img_cartoon = cv2.bitwise_and(img, img_edge)

    #一行三列，位置为1，imshow的颜色顺序RGB
    plt.subplot(131),plt.imshow(cimg)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # 一行三列，位置为2
    plt.subplot(132),plt.imshow(blur)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # 一行三列，位置为3
    plt.subplot(133),plt.imshow(img_edge,cmap = 'gray')#灰度图
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()