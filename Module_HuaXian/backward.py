# 反向传播，用到了学习率的指数衰减，让衰减过程加快；用到了正则化返乡传播
# 0导入模块
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import generateds
import forward

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01


def backward():
    # 列出x y y_的式子
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_c = generateds.generate()
    y = forward.forward(x, REGULARIZER)
    # 记录当前轮数，学习率指数衰减
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 300 / BATCH_SIZE, LEARNING_RATE_DECAY,
                                               staircase=True)

    # 定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    # 定义返乡传播方法，使用正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print("After %d steps,loss is %g" % (i, loss_v))

        # 画图：
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        # grid放的是36玩个点的坐标
        grid = np.c_[xx.ravel(), yy.ravel()]
        # probs就是求出来的y   360000,1
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)

        # 画出数据集中的点，标为红/蓝色
        plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
        # 画出等高线,probs也就是y值为0.5的点，连起来
        plt.contour(xx, yy, probs, levels=[.5])
        plt.show()


if __name__ == '__main__':
    backward()
