import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model, load_model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import cv2 as cv

inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()
embed_input = Input(shape=(1000,))

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

batch_size=10
def process_pic(pic):#pic已经是读出来的一张图片，返回图片的L数组 图片嵌入 和图片的a b数组
    Xtemp = []
    X_single=[]
    Xtemp = img_to_array(pic)
    Xtemp = np.resize(Xtemp, (512, 512, 3))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    Xtemp = cv.filter2D(Xtemp, -1, kernel=kernel)  # 锐化
    Xtemp = 1.0 / 255 * Xtemp #返回处理好的图片数组

    grayscaled_rgb = gray2rgb(rgb2gray(Xtemp))
    embed=create_inception_embedding(grayscaled_rgb)#图片嵌入

    lab_single=rgb2lab(Xtemp)#元图片数组从rgb转为lab
    # print(lab_single.shape) #512 512 3

    # print(lab_single[:, :, :, 0])
    # X_single=X_single.reshape(1,512,512)
    # X_single=lab_single[:, :, :, 0]
    X_single = lab_single[:, :, 0]
    X_single=X_single.reshape(X_single.shape+(1,))#图片L

    # Y_single=lab_single[:,:,:,1:] / 128#图片a b
    Y_single = lab_single[:, :, 1:] / 128
    return X_single,embed,Y_single

def generate_arrays_from_path(batch_size):#   # './Train/'
    Xtrain=[]
    Ytrain=[]
    embed_train=[]
    cnt=0
    for filename in os.listdir('./Train/'):
        x,embed,y=process_pic(load_img('./Train/'+filename))
        Xtrain.append(x)
        Ytrain.append(y)
        embed_train.append(embed)
        cnt+=1
        if cnt==batch_size:
            cnt=0
            # yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)
            yield ([Xtrain,embed_train],Ytrain)
            Xtrain=[]
            Ytrain=[]
            embed_train=[]