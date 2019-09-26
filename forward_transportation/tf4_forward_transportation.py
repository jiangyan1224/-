#与tf3不同的是，喂数据用的不是constant，
# 是placeholder,在run方法中用feed_dict喂数据，可以一次喂入多组数据

# none是指输入数据的组数，这里指先不确定组数
# x=tf.placeholder(tf.float32,shape=(None,2))
# 喂了三组数据
# sess.run(y,feed_dict={x:[[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5]]})

#coding:utf-8
#简单两层神经网络（全连接）
import tensorflow as tf
#placeholder可以作为 java 方法中的形参，用于定义过程，在方法执行时再赋予具体的值,
# 这里shape后第一个参数不是none是1，确定这次喂入一组数据，但是具体是什么数据要在后面赋值
x=tf.placeholder(tf.float32,shape=(1,2))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,w1)
y=tf.matmul(a,w2)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf4_forward_transportation is:\n", sess.run(y,feed_dict={x:[[0.7,0.5]]}))