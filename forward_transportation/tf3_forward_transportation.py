#coding:utf-8
#两层简单神经网络(全连接)
import tensorflow as tf
#定义输入和输出参数
x=tf.constant([[0.7,0.5]])
#第一层参数
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
#第二层参数
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)#y为最终结果

#用会话计算结果
with  tf.compat.v1.Session() as sess:
    #tf.initialize_all_variables，是预先对变量初始化，Tensorflow 的变量必须先初始化，然后才有值！
    init_op=tf.global_variables_initializer()
    #initialization variables
    sess.run(init_op)
    print("y in tf3_forward_transportation is:\n",sess.run(y))