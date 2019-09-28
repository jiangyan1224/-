# 预测酸奶日销量y x1 x2是影响日销量的因素
# 建模前，应预先采集的数据集有：每日的x1 x2和销量y_（即已知答案，最佳情况就是y=y_）
# 拟造数据集X Y_:y_=x1+x2 噪声：-0.05~+0.05
#(假设y和x1 x2的关系是x1+x2，用训练证明这个式子是否正确)

# coding:utf-8
# 0 导入模块，生成数据集
import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455
# 基于seed种子产生随机数
rdm = np.random.RandomState(SEED)
# 32组0-1之间的数据，每组两个特征；作为输入数据集
X = rdm.rand(32, 2)
# 每组数据取出来，求和，再加上随机噪声，构造标准答案
# rand()生成0-1，/10，生成0-0.1，减去0.05，变成-0.05~+0.05之间的随机数
Y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in X]

# 1定义神经网络的输入 参数 输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 2定义损失函数和反向传播方法，损失函数为MSE，反向传播方法为梯度下降
loss_mse = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 3生成会话，训练STEPS轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 20000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = (i * BATCH_SIZE) % 32 + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        # 每500轮，打印一次w1
        if i % 500 == 0:
            print("After %d training steps,w1 is:" % (i))
            print(sess.run(w1), "\n")
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training steps,loss is %g\n" % (i, total_loss))
    print("final w1 is:\n", sess.run(w1))
