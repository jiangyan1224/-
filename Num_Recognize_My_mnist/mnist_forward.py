#前向传播
#用到正则化
import tensorflow as tf
#要求输入的数据是784个像素点，每个像素点是0~1的浮点数（越接近0越黑，越接近1越白）以此作为数据输入
#输出是一个1X10的张量，10个值中最大的元素对应的索引号为原图片对应的数字答案
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

def get_weight(shape,regularizer):
    #从截断的正态分布中输出随机值。
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    #将内容加入正则化集合,反向传播中加入总loss
    if regularizer!=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    #返回一个具有shape形状的dtype类型的张量，所有元素都设置为零。
    b=tf.Variable(tf.zeros(shape))
    return b

def forward(x,regularizer):
    w1=get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b1=get_bias([LAYER1_NODE])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)

    w2=get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
    b2=get_bias([OUTPUT_NODE])
    y=tf.matmul(y1,w2)+b2
    return y