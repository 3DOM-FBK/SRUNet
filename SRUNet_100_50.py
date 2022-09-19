# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:37:09 2021

@author: salim
"""


# from model1 import *
from datetime import datetime

import glob
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import vstack, hstack
from numpy.random import randn, rand
from numpy.random import randint, permutation
import random
import os

from tensorflow.keras import applications
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np 
import os

import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
import tensorflow.keras.backend as K

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
dim=64  # 224
dimG=128
bands = 1


def custom_loss(y_true, y_pred):
   SSIML=tf.image.ssim(y_true,y_pred,max_val=150)
   loss1 = 2*(1-SSIML)
   loss22 = tf.keras.metrics.MAE(y_true,y_pred)
   loss23 = tf.reduce_mean(loss22, axis=-1)
   loss2 = tf.reduce_mean(loss23, axis=-1)
   return (loss1 + loss2)


def unet1(input_size):
    inputs = Input(input_size)  # 0
    # layers 1-3
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) # 1
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)  # 2
    pool1 = AveragePooling2D(pool_size=(2, 2))(conv1) # 3
    
    # layers 4-6
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)  # 4
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)  # 5
    pool2 = AveragePooling2D(pool_size=(2, 2))(conv2) # 6
 
    # layers 7-9
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2) # 7
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # 8
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # 9

    # layers 10-14
    up5 = UpSampling2D(size = (2,2), interpolation="bilinear")(conv4)  # 10
    merge5 = concatenate([up5,conv2], axis = 3)  # 11
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5) # 12
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) # 13
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) # 14

    # layers 15-19
    up6 = UpSampling2D(size = (2,2), interpolation="bilinear")(conv5) # 15
    merge6 = concatenate([up6,conv1], axis = 3)  # 16
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6) # 17
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6) # 18
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6) # 19
    
    # layers 20-26
    up7 = UpSampling2D(size = (2,2), interpolation="bilinear")(conv6)  # 20
    up_inputs = UpSampling2D(size = (2,2), interpolation="bilinear")(inputs)  # 21
    merge7 = concatenate([up7,up_inputs], axis = 3) # 22
    conv7 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7) # 23
    conv7 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) # 24
    conv7 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) # 25
    conv7 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) # 26

    conv8 = Conv2D(1, 1, activation = 'relu', padding = 'same',)(conv7)  # 27
    model = Model(inputs, conv8)  # 26

    model.compile(optimizer = Adam(lr = 1e-4), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    return model

cwd = os.getcwd()

IMG_DIM = (dim, dim)
input_shape1 = (dim, dim, 3)
input_shape = (dim, dim, 1)
model1 = unet1(input_shape)
model1.summary()
print('start***************************************')

vgg_16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape1)
# print('----- pls1 -------')
w1 =  vgg_16.layers[1].get_weights()
w2 = w1[0]
w2m = np.mean(w2,2)
w2m = np.reshape(w2m,(3,3,1,64))
w21 = w1[1]
w1[0] = w2m
w1[1] = w21
# copy layers of pretrained vgv256 to my model
model1.layers[1].set_weights(w1)
model1.layers[2].set_weights(vgg_16.layers[2].get_weights())
model1.layers[4].set_weights(vgg_16.layers[4].get_weights())
model1.layers[5].set_weights(vgg_16.layers[5].get_weights())
model1.layers[7].set_weights(vgg_16.layers[7].get_weights())
model1.layers[8].set_weights(vgg_16.layers[8].get_weights())
model1.layers[9].set_weights(vgg_16.layers[9].get_weights())

model1.compile(optimizer = Adam(lr = 1e-4), loss = custom_loss, metrics = ['RootMeanSquaredError'])
print('-------------------------------------------------curent path=',cwd)

print('----- read data1 -------')
dir_tr=cwd +'/Train_100_50/Input_im/'
dir_gt=cwd +'/Train_100_50/Label_im/'
train_files = glob.glob(os.path.join(dir_tr, '*.npy'))
lab_train_files = glob.glob(os.path.join(dir_gt, '*.npy'))
files_name = [fn.split('/')[-1].split('.npy')[0].strip() for fn in train_files]
print('------------------------lenth tr = ',str(len(train_files)))
# print(files_name)
N = len(files_name)

epochs1 = 1
batch_size1 = 50
train_imgs=zeros((batch_size1,dim,dim,1))
train_labels=zeros((batch_size1,dimG,dimG,1))

N1=np.uint16(N/batch_size1) #800
N2=batch_size1
start=datetime.now()
tr_Acc=np.zeros((300,2))
time_Tr=np.zeros((300,2))

MAE_list = []

MAE_min=9999999999.99
stop=0
i=-1
epochs_max=300
max_nb_min=5
nb_min=0
iter0=0
print('start init tr1 --------------------------------------------------------------')
while((i<epochs_max) and (stop==0)):
    i=i+1
    iter1=epochs1*(i+1)+iter0
    start1=datetime.now()
    rand_p=permutation(N)
    idi=0
    lossTr=0  
    for i21 in range(N1):
        for i22 in range(N2):
        
            i2=rand_p[idi]
            idi=idi+1
            train_imgs[i22,:,:,0]=np.load(dir_tr+files_name[i2][:]+'.npy')
            train_labels[i22,:,:,0]=np.load(dir_gt+files_name[i2][:]+'.npy')

        print('SRUNet_100_50  iteration number   ',iter1,' sub-iter = ',i21+1)
        history_model1 = model1.fit(train_imgs, train_labels,
                    epochs=epochs1, # shuffle=false,
                    batch_size=batch_size1) 
        
        a=history_model1.history['loss']
        lossTr=lossTr+a[0]
    tr_Acc[iter1-1,0]=iter1-1
    tr_Acc[iter1-1,1]=lossTr/N1
    stopTr=datetime.now()
    timeTr=stopTr-start1
    time_Tr[iter1-1,0]=iter1-1
    time_Tr[iter1-1,1]=timeTr.seconds
    
    if(tr_Acc[iter1-1,1]>MAE_min):
        nb_min=nb_min+1
    else:
        MAE_min = tr_Acc[iter1-1,1]
        nb_min = 0
        model1.save(cwd +'/SRUNet_100_50.h5')
        
    if(nb_min>max_nb_min):
        stop=1

    if(iter1==2):
        model1.compile(optimizer = Adam(lr = 5e-5), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==4):
        model1.compile(optimizer = Adam(lr = 2e-5), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==8):
        model1.compile(optimizer = Adam(lr = 1e-5), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==16):
        model1.compile(optimizer = Adam(lr = 5e-6), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==32):
        model1.compile(optimizer = Adam(lr = 2e-6), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==64):
        model1.compile(optimizer = Adam(lr = 1e-6), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==128):
        model1.compile(optimizer = Adam(lr = 5e-7), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==196):
        model1.compile(optimizer = Adam(lr = 2e-7), loss = custom_loss, metrics = ['RootMeanSquaredError'])
    if(iter1==256):
        model1.compile(optimizer = Adam(lr = 1e-7), loss = custom_loss, metrics = ['RootMeanSquaredError'])

    np.save(cwd +'/tr_Acc_SRUNet_100_50',tr_Acc)
    np.save(cwd +'/Tr_runtime_SRUNet_100_50',time_Tr)



