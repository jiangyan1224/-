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
# Get images

EPOCHS = 5
# X = []
# Xtemp=[]
# for filename in os.listdir('./Train/'):
#     Xtemp=img_to_array(load_img('./Train/'+filename))
#     Xtemp = np.resize(Xtemp, (512, 512, 3))
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
#     Xtemp = cv.filter2D(Xtemp, -1, kernel=kernel)  # 锐化
#     X.append(Xtemp)
#     # X.append(img_to_array(load_img('./Train/'+filename)))
# X = np.array(X, dtype=float)
# Xtrain= 1.0 / 255 *X


#Load weights
inception = InceptionResNetV2(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()
embed_input = Input(shape=(1000,))

#Encoder
encoder_input = Input(shape=(512, 512, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#Fusion
fusion_output = RepeatVector(64 * 64)(embed_input)
fusion_output = Reshape(([64, 64, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3)
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
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

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data##################
# batch_size = 10
# def image_a_b_gen(batch_size):
#     for batch in datagen.flow(Xtrain, batch_size=batch_size):
#         grayscaled_rgb = gray2rgb(rgb2gray(batch))#图片嵌入
#         # embed = create_inception_embedding(grayscaled_rgb)#图片嵌入
#         lab_batch = rgb2lab(batch)
#         X_batch = lab_batch[:,:,:,0]#图片L
#         # print(X_batch.shape)
#         X_batch = X_batch.reshape(X_batch.shape+(1,))#图片L
#         Y_batch = lab_batch[:,:,:,1:] / 128#图片a b
#         yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)
#######################################
batch_size=2
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
###############################
#Train model

weight_file='./model.h5'
if os.path.exists(weight_file):
    model=load_model(weight_file)
    print('model has loaded............')
else :
    print('first train')
    model.compile(optimizer='rmsprop', loss='mse')
# model.fit_generator(image_a_b_gen(batch_size), epochs=EPOCHS, steps_per_epoch=1)
model.fit_generator(generate_arrays_from_path(batch_size),epochs=EPOCHS,steps_per_epoch=1)
model.save(weight_file)

model=load_model(weight_file)
color_me = []
temp=[]
for filename in os.listdir('./Test/'):
    temp=img_to_array(load_img('./Test/' + filename))
    temp = np.resize(temp, (512, 512, 3))
    color_me.append(temp)

color_me = np.array(color_me, dtype=float)
gray_me = gray2rgb(rgb2gray(1.0/255*color_me))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((512, 512, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))