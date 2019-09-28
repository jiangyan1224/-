#滑动平均/影子值：记录了每个参数一段时间内过往值的平均，像影子，参数变化，影子缓慢追随
# 针对所有参数w和b
#影子初值＝参数初值
#参数分别为　滑动平均衰减率和当前轮数
# ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
#下面这句每次运行，都会将所有待训练的参数汇总成一个链表
# ema_op=ema.apply(tf.trainable_variables())
#在工程应用中，常把计算滑动平均和训练过程放到一起，一起构成训练节点
# with tf.control_dependencies([train_step,ema_op])
#     train_op=tf.no_op(name='train')
#ema.average(参数名)可以返回某个参数的滑动平均值
#coding:utf-8
import tensorflow as tf

#1定义变量和滑动平均类
#定义一个32位浮点变量，初始值为0.0，这个代码就是不断更新w1参数，优化w1参数，滑动平均做了w1的影子
w1=tf.Variable(tf.constant(0.0,dtype=tf.float32))
#定义NN的迭代论述，即计数器，初始为0，不可优化
global_step=tf.Variable(0,trainable=False)

#实例化滑动平均类，删减率为0.99
MOVING_AVERAGE=0.99
ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE,global_step)
#每次运行，都会将所有待训练的参数汇总成一个链表，返回为滑动平均节点
ema_op=ema.apply(tf.trainable_variables())

#2查看不同迭代中变量的变化
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    #打印出当前参数w1和w1的滑动平均值
    print(sess.run([w1,ema.average(w1)]))

    #把参数w1值赋值为1，打印出赋值之后的参数w1和w1的滑动平均值
    sess.run(tf.assign(w1,1))
    #将所有待训练的参数汇总成一个链表，运行返回的滑动平均节点
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))

    #再次更新w1的值，也改变global_step的值，模拟经过100轮后，参数w1变为10
    sess.run(tf.assign(global_step,100))
    sess.run(tf.assign(w1,10))
    #将所有待训练的参数汇总成一个链表，运行返回的滑动平均节点
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))

    #每次执行sess.run(),都会更新一次w1的滑动平均值
    #改变滑动平均衰减率MOVING_AVERAGE_DECAY，影子追随参数w1的速度也会发生变化
    sess.run(ema_op)
    print(sess.run([w1,ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))