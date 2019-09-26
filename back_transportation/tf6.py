#coding:utf-8
#导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
#一次喂入神经网络BATCH_SIZE组数据，不能过大
BATCH_SIZE=8
seed=23455
#基于seed种子产生随机数
rng=np.random.RandomState(seed)
#随机数返回32列2行的矩阵，表示32组数据，每组两个特征；作为输入数据集
X=rng.rand(32,2)
#从X这个32行2列的矩阵中取出一行，判断如果和小于1，<判断为真，给Y赋值1，否则赋值0
#作为输入数据集的标签/正确答案
Y=[[int(x0+x1<1)] for (x0,x1) in X]
print("X:\n",X)
print("Y:\n",Y)

#1定义神经网络的输入 参数和输出 定义前向传播过程
#不知道一共可以拿到多少组数据
x=tf.placeholder(tf.float32,shape=(None,2))
#每组数据的标准答案
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

#2定义损失函数和反向传播方法
#用均方误差定义损失函数loss；
# 用梯度下降实现训练过程，学习率设为0.001 ；或者使用MomentumOptimizer/AdadeltaOptimizer优化方法
#优化的目的是尽可能让y误差减小，接近ｙ＿
loss=tf.reduce_mean(tf.square(y-y_))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step=tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
# train_step=tf.train.AdadeltaOptimizer(0.001).minimize(loss)

#生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #输出目前优化之前 未经训练的参数值
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))
    print("\n")

    #训练模型,训练STPES轮
    STEPS=3000
    for i in range(3000):
        start=(i*BATCH_SIZE)%32
        end=start+BATCH_SIZE
        #每一轮从X Y中，从start到end的切片数据组，X Y相对应，喂入模型
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        #每500轮，打印一次loss值
        if i % 500== 0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training steps,loss on all data is %g" % (i,total_loss))

    #输出训练后的参数取值
    print("\n")
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))


