#生成数据集
#coding:utf-8
#0导入模块，生成模拟数据集
import numpy as np
import matplotlib.pyplot as plt
seed=2

def generate():
    #基于seed产生随机数
    rdm=np.random.RandomState(seed)
    X=rdm.randn(300,2)
    Y_=[int(x0*x0+x1*x1<2) for (x0,x1) in X]
    Y_c=[['red' if y_ else 'blue'] for y_ in Y_]

    #对数据集X和标签Y_整理形状
    X=np.vstack(X).reshape(-1,2)
    Y_=np.vstack(Y_).reshape(-1,1)

    # print(X)
    # print('------------\n')
    # print(Y_)
    # print('------------\n')
    # print(Y_c)

    return X, Y_, Y_c