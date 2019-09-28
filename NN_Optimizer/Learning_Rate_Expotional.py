#指数衰减学习率：学习率不会一直都是人为设置的初值，而是会随着训练的轮数进行衰减调整
# 设置一个计数器global_step，记录当前是第几次训练，trainable=False意为这个变量不是训练参数，不要对其训练
# global_step=tf.Variable(0,trainable=False)
# 参数分别为：人为设置的学习率初值 当前第几轮训练 多少轮BATCH_SIZE更新一次学习率=总样本数/BATCH_SIZE，学习率衰减率，让学习率梯度衰减还是曲线衰减
# learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)

#coding:utf-8
#设损失函数为(w+1)^2,w初值为10，反向传播求最小loss对应的w值
#使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得更好的收敛度
import tensorflow as tf
#最初学习率
LEARNING_RATE_BASE=0.1
#学习率衰减率
LEARNING_RATE_DECAY=0.99
#多少轮BATCH_SIZE后更新一次学习率
LEARNING_RATE_STEP=1
#当前是第几轮训练，初值为0，非训练参数
global_step=tf.Variable(0,trainable=False)

#定义指数衰减学习率：
learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase=True)
#定义优化参数，初值为5：
w=tf.Variable(tf.constant(5,dtype=tf.float32))
#定义损失函数
loss=tf.square(w+1)

#定义反向传播方法：
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

#生成会话，训练40轮：
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(40):
        sess.run(train_step)
        w_val=sess.run(w)
        loss_val=sess.run(loss)
        print("After %d steps,w is %f  ,loss is %f" %(i,w_val,loss_val))