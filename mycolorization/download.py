#-*-coding: utf-8 -*-
'''
使用beautifulsoup下载图片
1、使用urllib.request 下载到网页内容
2、使用beautifulsoup匹配到所有的图片地址
3、指定文件路径
4、调用urllib.request.urlretrieve 下载图片
'''
import urllib.request

import untangle as untangle
from bs4 import BeautifulSoup
import numpy as np
import cv2

count=0
maxsize=512
for i in range(10000):
    stringreturn = urllib.request.urlopen(
        "http://safebooru.org/index.php?page=dapi&s=post&q=index&tags=1girl%20solo&pid=" + str(i + 3000))
    xmlreturn=untangle.parse(stringreturn)
    for post in xmlreturn.posts.post:
        # print(post["sample_url"])
        imgurl=post["sample_url"]
        print(imgurl)

        if("png" in imgurl)or("jpg" in imgurl):
            resp=urllib.request.urlopen(imgurl)#打开图片的url
            image=np.asarray(bytearray(resp.read()),dtype="uint8")#转为bytearray数组
            image=cv2.imdecode(image,cv2.IMREAD_COLOR)#把图片数据转为图片，以彩色模式读入
            height,width=image.shape[:2]#取彩色图片的高、宽

            if height>width:
                scalefactor=(maxsize*1.0)/width
                res=cv2.resize(image,(int(width*scalefactor),int(height*scalefactor)),
                               interpolation=cv2.INTER_CUBIC)#基于4x4像素邻域的3次插值法
                cropped=res[0:maxsize,0:maxsize]#宽 高各取0-maxsize像素点

            if width>height:
                scalefactor=(maxsize*1.0)/height
                res = cv2.resize(image, (int(width * scalefactor), int(height * scalefactor)),
                                 interpolation=cv2.INTER_CUBIC)  # 基于4x4像素邻域的3次插值法
                center_x=int(round(width*scalefactor*0.5))#四舍六入 五留双
                print(center_x)
                cropped=res[0:maxsize,center_x-maxsize//2:center_x+maxsize//2]

            count+=1
            cv2.imwrite("train/"+str(count)+".jpg",cropped)


