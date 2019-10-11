import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np
#整个程序循环的间隔时间为5秒
TEST_INTERVAL_SECS=5

def test(mnist):
    #tf.Graph()复现计算图
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[
            mnist.test.num_examples,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS
        ])
        y_=tf.placeholder(tf.float32,[None,mnist_lenet5_forward.OUTPUT_NODE])
        y=mnist_lenet5_forward.forward(x,False,None)

        #滑动平均里面，模型的恢复
        ema=tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)

        #正确率的式子
        #利用tf.argmax()按行求出真实值y_、预测值y最大值的下标，
        #用tf.equal()求出真实值和预测值相等的数量，也就是预测结果正确的数量，tf.argmax()和tf.equal()一般是结合着用。
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accurancy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        while True:
            with tf.Session() as sess:
                #加载ckpt，把滑动平均值赋给各个参数
                ckpt=tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
                #如果有对应模型，恢复模型到当前会话
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #恢复global_step值
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    reshaped_x=np.reshape(mnist.test.images,(
                        mnist.test.num_examples,
                        mnist_lenet5_forward.IMAGE_SIZE,
                        mnist_lenet5_forward.IMAGE_SIZE,
                        mnist_lenet5_forward.NUM_CHANNELS
                    ))
                    #执行正确率的计算
                    accurancy_score=sess.run(accurancy,feed_dict={x:reshaped_x,y_:mnist.test.labels})
                    print("After %s training step, test accuracy is %g" %(global_step,accurancy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    test(mnist)

if __name__=="__main__":
    main()
