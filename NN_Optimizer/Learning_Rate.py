# 学习率：learning_rate:每次参数更新的幅度
# wn+1=wn-learning_rate*损失函数的梯度
# coding:utf-8
# 设损失函数 loss=(w+1)^2，令w初值为常数5，反向传播求解最优w，即求最小loss对应的w值
# 看看代码能不能找到损失函数的最小值处，即w为-1，loss为0
import tensorflow as tf

# 定义待优化参数w初值为5
w = tf.Variable(tf.constant(5, dtype=tf.float32))
# 定义损失函数loss
loss = tf.square(w + 1)
# 定义反向传播方法：
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 生成会话，训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %d steps: w is %f,   loss is %f" % (i, w_val, loss_val))
#PS:如果这里的学习率不设置为0.2,设置成过大的1：会让loss振荡不收敛，w也随着振荡，而到不了中间的最小值
# 反之，如果学习率设置得过小，loss收敛的速度会非常缓慢，需要更多次训练才能达到效果