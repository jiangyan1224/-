# 反向传播

# backward和test文件要改一下图片和标签获取的接口
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
###########################################
import mnist_generateds

###########################################

# 一次喂入200张图片，也就是200个INPUT_NODE
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
###########################################
train_num_examples = 60000


##############################################


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    # 这里不用均方误差而是用交叉熵作为损失函数一部分，因为这里是一个分类问题
    # 均方误差更适合做回归问题，而且交叉熵比均方误差能更快到达局部最优解
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 学习率指数衰减
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        ############################################
        # 多少轮学习率更新一次，总样本数/BATCH_SIZE,这里需要手动给出总样本数为60000
        train_num_examples / BATCH_SIZE,
        # mnist.train.num_examples / BATCH_SIZE,
        ##############################################
        LEARNING_RATE_DECAY,
        staircase=True)

    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 每次运行，都会将所有待训练的参数汇总成一个链表，返回为滑动平均节点
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()
    #######################################################
    img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)
    ########################################################

    # 会话
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # 在这里加载ckpt断点，如果已经有训练过的模型，将其恢复到当前会话，
        # 给所有的w b赋值为保存在ckpt里的值，实现断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        #######################################################
        # 为提高效率，调用了线程协调器
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ########################################################

        for i in range(STEPS):
            # 按批次训练，一次BATCH_SIZE个图片和标签
            ##########################################################
            # 需要手动通过generateds文件的接口读取随机的batch_size组数据
            xs, ys = sess.run([img_batch, label_batch])
            # xs, ys = mnist.train.next_batch(BATCH_SIZE)
            ##########################################################
            # 这里的 _ 对应train_op的输出，空出一个变量位，节省空间
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d steps loss on training batch is %g" % (step, loss_value))
                # 保存会话和当前模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        ####################################
        # 关闭线程协调器
        coord.request_stop()
        coord.join(threads)
        ########################################


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()



