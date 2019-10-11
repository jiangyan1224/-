# backward和test文件要改一下图片和标签获取的接口
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
#整个程序循环的间隔时间为5秒
TEST_INTERVAL_SECS=5

################################
import mnist_generateds
#手动说明测试总样本数
TEST_NUM=10000
################################

def test(mnist):
    #tf.Graph()复现计算图
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
        y_=tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
        y=mnist_forward.forward(x,None)

        #滑动平均里面，模型的恢复
        ema=tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)

        #正确率的式子
        #利用tf.argmax()按行求出真实值y_、预测值y最大值的下标，
        #用tf.equal()求出真实值和预测值相等的数量，也就是预测结果正确的数量，tf.argmax()和tf.equal()一般是结合着用。
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accurancy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        ##############################################
        #所有测试集，10000张图片，不是训练是测试，所以是false
        img_batch,label_batch=mnist_generateds.get_tfrecord(TEST_NUM,isTrain=False)
        ##################################################

        while True:
            with tf.Session() as sess:
                #加载ckpt，把滑动平均值赋给各个参数
                ckpt=tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                #如果有对应模型，恢复模型到当前会话
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #恢复global_step值
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    ##########################################
                    #打开线程协调器
                    coord=tf.train.Coordinator()
                    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
                    #批量获取图片和标签
                    xs,ys=sess.run(img_batch,label_batch)

                    #执行正确率的计算
                    # accurancy_score=sess.run(accurancy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
                    accurancy_score=sess.run(accurancy,feed_dict={x:xs,y_:ys})
                    print("After %s training step, test accuracy is %g" %(global_step,accurancy_score))

                    #关闭线程协调器
                    coord.request_stop()
                    coord.join(threads)
                    ####################################################
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    test(mnist)

if __name__=="__main__":
    main()
