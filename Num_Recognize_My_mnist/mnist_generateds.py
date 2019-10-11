#制作数据集，自动将自己的真实图片制作成数据集进行读取，backward和test文件要改一下图片和标签获取的接口

import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path='./mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path='./mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train='./data/mnist_train.tfrecords'
image_test_path='./mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path='./mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test='./data/mnist_test.tfrecords'
data_path='./data'
resize_height=28
resize_width=28


def read_tfRecord(tfRecord_path):
    #获取所有tfRecord文件的路径，这里只有一个tfRecord文件
    filename_queue=tf.train.string_input_producer([tfRecord_path])
    reader=tf.TFRecordReader()
    #把每一个样本读出来
    _,serialized_example=reader.read(filename_queue)
    #，解序列化
    features=tf.parse_example(serialized_example,features={
        #这里的[10]是指标签是10分类
        'label':tf.FixedLenFeature([10],tf.int64),
        'img_raw':tf.FixedLenFeature([],tf.string)
    })
    #将img_raw转换为8位无符号整型
    img=tf.decode_raw(features['img_raw'],tf.uint8)
    #把形状变为1x784
    img.set_shape([784])
    #再变成浮点数，变成0~1之间
    img=tf.cast(img,tf.float32)*(1./255)
    #把标签变为浮点数
    label=tf.cast(features['label'],tf.float32)
    return img,label

#批量读取训练集/训练集代码，num指一次读取多少组数据，true指读取测试集
def get_tfrecord(num,isTrain=True):
    if isTrain:
        tfRecord_path=tfRecord_train
    else:
        tfRecord_path=tfRecord_test
    #读取tfRecord文件里面的图片1x784，和标签1x10
    img,label=read_tfRecord(tfRecord_path)

    #从总样本数中顺序取出capacity组数据，打乱顺序，每次输出batch_size组
    #如果capacity<min_after_dequeue，会从总样本数中取数据，填满capacity
    #整个过程使用两个线程
    img_batch,label_batch=tf.train.shuffle_batch([img,label],
                                                 batch_size=num,
                                                 num_threads=2,
                                                 capacity=1000,
                                                 min_after_dequeue=700)
    #返回随机取出的batch_size组数据
    return img_batch,label_batch

#写入tfRecords文件
def write_tfRecord(tfRecordName,image_path,label_path):
    #创建一个writer
    writer=tf.python_io.TFRecordWriter(tfRecordName)
    #创建一个计数器
    num_pic=0
    #以读的形式打开标签文件，标签文件是一个txt文件，每一行由图片名和标签组成，中间空格隔开
    f=open(label_path,'r')
    contents=f.readlines()
    f.close()

    for content in (contents):
        value=content.split()
        #image_path是图片所在文件夹的路径
        img_path=image_path+value[0]
        #img_path是更具体的图片路径
        img=Image.open(img_path)
        img_raw=img.tobytes()

        labels=[0]*10
        #把labels数组中对应的标签位 设为1
        labels[int(value[1])]=1

        #封装example
        example=tf.train.Example(features=tf.train.Features(feature={
            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
        }))

        #把example进行序列化
        writer.write(example.SerializeToString())
        num_pic+=1
        print("the number of pictures:",num_pic)

    writer.close()
    print("write tfrecord successful")

def generate_tfRecord():
    #判断保存路径是否存在
    isExists=os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The dirctory is successfully created")
    else:
        print("directory already exists")
        #把训练集的图片和标签生成名为tfRecord_train的tfRecords文件
        write_tfRecord(tfRecord_train,image_train_path,label_train_path)
        write_tfRecord(tfRecord_test,image_test_path,label_test_path)

def main():
    generate_tfRecord()

if __name__=='__main__':
    main()