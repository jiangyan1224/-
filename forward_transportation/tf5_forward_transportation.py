#placeholder喂多组数据

#coding:utf-8
#简单两层神经网络（全连接）
import tensorflow as tf
#placeholder可以作为 java 方法中的形参，用于定义过程，在方法执行时再赋予具体的值,
# 这里shape后第一个参数是none,具体多少组数据，什么数据，都在后面定义
x=tf.placeholder(tf.float32,shape=(None,2))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #y根据四组数据算出4组y
    print("y in tf5_forward_transportation is:\n", sess.run(y,feed_dict=
    {x:[[0.7,0.5],[0.2,0.3],[0.3,0.4],[0.4,0.5]]}))
    #查看随即生成的w1 w2
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))