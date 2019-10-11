# 这个项目和MNIST_num_recognize的区别在于，增加了这个app.py文件，用于把真实的图片转化为可以直接喂入神经网络的张量
# 因为训练模型不变，所以forward backward test文件不变
import tensorflow as tf
import numpy as np
from PIL import Image
from pip._vendor.distlib.compat import raw_input

import mnist_backward
import mnist_forward


# 得到预处理后的1x784的图片信息，通过ckpt使用神经网络，得到识别的数字
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        # preValue是网络识别出的值
        preValue = tf.argmax(y, 1)

        # 滑动平均里面，模型的恢复     #实例化带有滑动平均值的saver
        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_averages_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_averages_restore)

        with tf.Session() as sess:
            # 加载ckpt，把滑动平均值赋给各个参数
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            # 如果有对应模型，恢复模型到当前会话
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                # sess.run()计算识别出的值
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


# 图片预处理：
def pre_pic(picName):
    # 打开原始图片
    img = Image.open(picName)
    # 读入图片路径，转化为数组(现有的RGB图每个像素点的值是0-255)
    # 将img resize为符合网络输入要求的大小；Image.ANTIALIAS值用消除锯齿的方法resize
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # reIm.convert('L')：将reIm变为灰度图，符合网络对输入图片的颜色要求
    # np.array把reIm转化为矩阵的形式赋给im_arr
    im_arr = np.array(reIm.convert('L'))

    # tensorflow手写数字识别的图片模型要求输入的图片是黑底白字，但是实际给的图片灰度化之后是白底黑字
    # 所以要给图片反色
    threshold = 50  # 阈值
    for i in range(28):
        for j in range(28):
            # 像素点变为255减去原像素点，得到互补的反色
            im_arr[i][j] = 255 - im_arr[i][j]
            # 如果小于阈值，置0，纯黑色
            if im_arr[i][j] < threshold:
                im_arr[i][j] = 0
            # 如果大于阈值，置255，纯白色，让图片只有纯黑纯白，减少图片中的噪声干扰
            else:
                im_arr[i][j] = 255

    # 把反色后的图片reshape为1x784的张量
    nm_arr = im_arr.reshape([1, 784])
    # 现有的RGB图每个像素点的值是0-255，但是网络要求输入的像素点是0~1的浮点数
    # 先从整数变为float32
    nm_arr = nm_arr.astype(np.float32)
    # 再从0-255变为0-1
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready


def application():
    # testNum指的是这次验证的图片个数，input方法读入数字
    testNum = input("input the number of test pictures:")
    testNum = int(testNum)
    for i in range(testNum):
        # 获取图片路径，raw_input方法读入字符串
        testPic = raw_input("the path of test picture:")
        # 将图片转化为数组
        testPicArr = pre_pic(testPic)
        # 将数组喂入神经网络模型，得到识别出来的数
        preValue = restore_model(testPicArr)
        print("the prediction number is: ", preValue)


def main():
    application()


if __name__ == '__main__':
    main()
