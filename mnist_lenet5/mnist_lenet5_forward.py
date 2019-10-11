import tensorflow as tf
#图片横向 纵向的边长
IMAGE_SIZE=28
#灰度图 通道数1
NUM_CHANNELS=1
#第一层卷积核大小是5
CONV1_SIZE=5
#第一层使用了32个卷积核
CONV1_KERNEL_NUM=32

CONV2_SIZE=5
CONV2_KERNEL_NUM=64
#第一层全连接网络有512个神经元
FC_SIZE=512
#第二层全连接网络有10个神经元
OUTPUT_NODE=10

def get_weight(shape,regularizer):
    #生成去掉过大偏离点的正态分布随机数
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    #正则化权重
    if regularizer!=None:tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    #初始值为0
    b=tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    #求卷积
    #输入描述 卷积核描述 滑动步长（中间两个，行步长列步长均为1） 使用0填充
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    #最大池化函数
    #输入描述 池化核大小（中间两个 2x2） 池化核步长（中间两个，行步长列步长均为2） 使用0填充
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def forward(x,train,regularizer):
    #卷积核描述：行分辨率 列分辨率 通道数 核个数
    conv1_w=get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)
    conv1_b=get_bias([CONV1_KERNEL_NUM])
    #中间的卷积结果
    conv1=conv2d(x,conv1_w)
    #激活函数处理
    relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    #最大池化
    pool1=max_pool_2x2(relu1)

    #第二层卷积核的深度等于上一层卷积核的个数
    conv2_w=get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
    conv2_b=get_bias([CONV2_KERNEL_NUM])
    #中间的卷积结果
    conv2=conv2d(pool1,conv2_w)
    #激活函数处理
    relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    #最大池化
    pool2=max_pool_2x2(relu2)

    #获得pool2的维度，存入list中
    pool_shape=pool2.get_shape().as_list()
    #pool_shape[1] [2] [3]分别表示pool的长度 宽度 深度 相乘得到所有特征点的个数
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3] #7 7 64
    # print(pool_shape[0],pool_shape[1],pool_shape[2],pool_shape[3])
    #pool_shape[0]表示一个BATCH的值；reshaped得到BATCH行数据，二维数据，喂入全连接网络
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

    fc1_w=get_weight([nodes,FC_SIZE],regularizer)
    fc1_b=get_bias([FC_SIZE])
    fc1=tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    #如果是训练阶段，对该层输出进行50%的dropout
    if train: fc1=tf.nn.dropout(fc1,0.5)

    fc2_w=get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
    fc2_b=get_bias([OUTPUT_NODE])
    y=tf.matmul(fc1,fc2_w)+fc2_b

    return y