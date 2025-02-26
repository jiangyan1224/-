#前向传播 用到了正则化
#coding:utf-8
import tensorflow as tf
#定义参数
def get_weight(shape,regularizer):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))#将内容加入正则化集合
    return w
def get_bias(shape):
    b=tf.Variable(tf.constant(0.01,shape=shape))
    return b
#前向传播过程 输出
def forward(x,regularizer):
    w1=get_weight([2,11],regularizer)
    b1=get_bias([11])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)

    w2=get_weight([11,1],regularizer)
    b2=get_bias([1])
    y=tf.matmul(y1,w2)+b2

    return y
