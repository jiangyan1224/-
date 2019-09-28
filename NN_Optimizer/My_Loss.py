# 在Loss.py中,假设y y_相等为最佳情况，但是实际上如果y>y_损失成本；反之损失利润
# 如：成本COST为1，利润PROFIT为9，那么希望预测函数y往多了预测比较好
#
# 自定义损失函数：loss(y_,y)=f1(y_,y)+f2(y_,y)+.....+fn(y_,y)
# 预测y少了：f(y_,y)=PROFIT*(y_-y)
# 预测y多了：f(y_,y)=COST*(y-y_)
# 下面这个式子类似于java的三元表达式, reduce_sum把所有的损失求和
# loss=tf.reduce_sum(tf.where(tf.grater(y,y_),COST*(y-y_),PROFIT*(y_-y)))

#coding:utf-8
#0导入模块
import tensorflow as tf
import numpy as np
BATCH_SIZE=8
SEED=23455
COST=1
PROFIT=9

#1生成数据集：
rdm=np.random.RandomState(SEED)
X=rdm.rand(32,2)
Y=[[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

#2定义输入 输出 参数 前向传播过程
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))
w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y=tf.matmul(x,w1)

#3定义损失函数和反向传播方法
loss=tf.reduce_sum(tf.where(tf.greater(y,y_),(y-y_)*COST,(y_-y)*PROFIT))
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#4生成会话，训练STEPS轮：
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    STEPS=20000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=(i*BATCH_SIZE)%32+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 500 == 0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training steps,w1 is:" %(i))
            print(sess.run(w1),"\n")
            print("After %d training steps,loss is: %g\n" %(i,total_loss))
    print("final w1 is:\n", sess.run(w1))
