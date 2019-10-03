#正则化可以缓解过度拟合，在loss总引入模型复杂度指标，给w加权值，弱化了数据的噪声，一般不正则化b
#loss=loss(y与y_)+REGULARIZER*loss(w)
#loss=模型中所有参数的损失函数，如交叉熵、均方误差 + REGULARIZER给出参数w在总loss中的比例，即正则化的权重 * loss(需要正则化的参数)

#loss(w)有两种计算方法：loss(w)=tf.contrib.layers.l1_regularizer(REGULARIZER)(w)
# loss(w)=tf.contrib.layers.l2_regularizer(REGULARIZER)(w)

# tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
# loss=cem+tf.add_n(tf.get_collection('losses'))

# 直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
# X[0,:]是指第0行所有数据
#coding:utf-8
#0导入模块，生成模拟数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE=30
seed=2
#基于seed产生随机数
rdm=np.random.RandomState(seed)
#随机数返回300行2列的矩阵，表示300组坐标作为输入数据集
X=rdm.randn(300,2)
# print(X)
#从X中取出一行，判断如果坐标平方和<2，给Y赋值1，否则为0，作为输入数据集的标签/正确答案
Y_=[int(x0*x0+x1*x1<2) for (x0,x1) in X]
#遍历Y中每个元素，如果为1赋值red，否则blue
Y_c=[['red' if y else 'blue'] for y in Y_]
#对X Y_进行shape处理，第一个元素为-1，随第二个参数计算得到，第二个元素表示多少列
X=np.vstack(X).reshape(-1,2)
Y_=np.vstack(Y_).reshape(-1,1)
print(X)
print(Y_)
print(Y_c)


#用plt.scatter画出数据集各行中第0列和第1列元素的点，即各行的(x0,x1)，用各行的Y_c对应值代表颜色
# 直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
# X[0,:]是指第0行所有数据
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
# print(np.squeeze(Y_c))np.squeeze(Y_c)去维度
plt.show();

#定义输入 参数 输出 前向传播过程
#定义一个生成指定shape和正则化权重的w参数的函数
def get_weight(shape,regularizer):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))#将内容加入集合
    return w
#定义一个生成b的函数
def get_bias(shape):
    #初值为0.01
    b=tf.Variable(tf.constant(0.01,shape=shape))
    return b
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

w1=get_weight([2,11],0.01)
b1=get_bias([11])
# tf.nn.relu()激活函数，返回与参数same type的张量Tensor
# tf.nn.relu()函数的目的是，将输入小于0的值幅值为0，输入大于0的值不变。
y1=tf.nn.relu(tf.matmul(x,w1)+b1)

w2=get_weight([11,1],0.01)
b2=get_bias([1])
#因为这一层是输出层，直接输出，输出层不过激活函数
y=tf.matmul(y1,w2)+b2

#定义损失函数
loss_mse=tf.reduce_mean(tf.square(y_-y))
# 均方误差的损失函数加上每一个正则化w的损失
loss_total=loss_mse+tf.add_n(tf.get_collection('losses'))

#A.定义反向传播方法：不含正则化：
train_step=tf.train.GradientDescentOptimizer(0.0001).minimize(loss_mse)
#会话运算
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=40000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%300
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%2000==0:
            loss_mse_v=sess.run(loss_mse,feed_dict={x:X,y_:Y_})
            print("After %d training steps,loss is %g" %(i,loss_mse_v))

    #xx在-3到3之间，以步长0.01，yy在-3到3之间以步长0.01，生成二维网格坐标点
    print('-------------------\n')
    xx,yy=np.mgrid[-3:3:.01,-3:3:.01]
    # print(xx)
    #将xx yy拉直，合并成一个2列的矩阵，得到一个网格坐标点的集合
    # grid放的是36玩个点的坐标
    grid=np.c_[xx.ravel(),yy.ravel()]#xx.ravel()函数是将多维数组转化为一维，np.c_[]将两个数组做融合
    # print(grid)
    #将网格点喂入神经网络，probs为输出
    # probs就是求出来的y
    probs=sess.run(y,feed_dict={x:grid})
    print(probs.shape)#360000,1
    # probs的shape调整成xx的样子
    probs=probs.reshape(xx.shape)
    print(probs)
    print(probs.shape)
    print("w1:\n",sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))
    # 用plt.scatter画出数据集各行中第0列和第1列元素的点，
    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()

#B.定义反向传播方法：含正则化：
train_step=tf.train.GradientDescentOptimizer(0.0001).minimize(loss_total)
#会话运算
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=40000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%300
        end=start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%2000==0:
            loss_mse_v=sess.run(loss_mse,feed_dict={x:X,y_:Y_})
            print("After %d training steps,loss is %g" %(i,loss_mse_v))

    #xx在-3到3之间，以步长0.01，yy在-3到3之间以步长0.01，生成二维网格坐标点
    print('-------------------\n')
    xx,yy=np.mgrid[-3:3.01,-3:3.01]
    # print(xx)
    #将xx yy拉直，合并成一个2列的矩阵，得到一个网格坐标点的集合
    grid=np.c_[xx.ravel(),yy.ravel()]#xx.ravel()函数是将多维数组转化为一维，np.c_[]将两个数组做融合
    print(grid)
    #将网格点喂入神经网络，probs为输出
    probs=sess.run(y,feed_dict={x:grid})
    # probs的shape调整成xx的样子
    probs=probs.reshape(xx.shape)
    print("w1:\n",sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))

    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()

