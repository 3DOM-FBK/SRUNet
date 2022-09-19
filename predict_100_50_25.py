# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:37:09 2021

@author: salim
"""

# example UNET on Pet dataset

# from model1 import *
# reset
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import glob
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import vstack, hstack
from numpy.random import randn, rand
from numpy.random import randint, permutation
import random
import os
# import matplotlib.pyplot as plt
import math
# from scipy import interpolate
import cv2
import copy

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras.utils import plot_model

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
# tf.executing_eagerly()
# tf.compat.v1.Session
dim=64  # 224
dimG=128
save_im=0  # 0 = no save image result, 1= save the result (predicted image) in format npy
full_image=0 # 0 = predict only 25% as on the paper, 1= predict the full image

def custom_loss(y_true, y_pred):
   SSIML=tf.image.ssim(y_true,y_pred,max_val=150)
   # loss = math_ops.mean(diff, axis=1) #mean over last dimension
   loss1 = 2*(1-SSIML)
   loss22 = tf.keras.metrics.MAE(y_true,y_pred)
   loss23 = tf.reduce_mean(loss22, axis=-1)
   loss2 = tf.reduce_mean(loss23, axis=-1)
   return (loss1 + loss2)

def psnr(img1, img2):
    mse = np.mean( (img1.astype("float") - img2.astype("float")) ** 2 )
    # print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



def mse(imageA, imageB, bands):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def rmse(imageA, imageB, bands):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
	err = np.sqrt(err)
	return err
	
	# return the MSE, the lower the error, the more "similar"
	# the two im
# div=4
bands = 1    
cwd = os.getcwd()




IMG_DIM = (dim, dim)
input_shape = (dim, dim, 1)

print('********** start **********************')
# first part> up from 1m to 50 com
model1=load_model(cwd +'/SRUNet_100_50.h5', custom_objects={'custom_loss': custom_loss})


print('----- read data -------')


## data1 LiDAR
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GT000 =  np.double(np.load(cwd+'/SR_test_data/data1_LiDAR_25cm.npy'))
# ima_2500 =  np.double(np.load(cwd+'/SR_test_data/data1_LiDAR_25cm.npy'))
# name_file='data1_LiDAR_25cm'
# sz0001=GT000.shape[0]; # height
# sz0002=GT000.shape[1]; # width
# if(save_im==1):
#     pred_full=np.zeros((sz0001,sz0002))
#     imagt_full=np.zeros((sz0001,sz0002))
# print('sz0001  ',sz0001,'  sz0002 ',sz0002)

# GT25=GT000[:,0:sz0002-2]   # -2 to be devided by 4
# ima_25=ima_2500[:,0:sz0002-2]  # -2 to be devided by 4
# sz001=GT25.shape[0]; # height
# sz002=GT25.shape[1]; # width
# print('sz001  ',sz001,'  sz002 ',sz002)

# sz01=int(sz001/2)
# sz02=int(sz002/2)
# print('sz01  ',sz01,'  sz02 ',sz02)
# GT50 = cv2.resize(GT25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
# ima_50 = cv2.resize(ima_25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)


## data1 Photogrammetry
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GT000 =  np.double(np.load(cwd+'/SR_test_data/data1_LiDAR_25cm.npy'))
# ima_2500 =  np.double(np.load(cwd+'/SR_test_data/data1_Photo_25cm.npy'))
# name_file='data1_LiDAR_25cm'
# sz0001=GT000.shape[0]; # height
# sz0002=GT000.shape[1]; # width
# if(save_im==1):
#     pred_full=np.zeros((sz0001,sz0002))
#     imagt_full=np.zeros((sz0001,sz0002))
# print('sz0001  ',sz0001,'  sz0002 ',sz0002)

# GT25=GT000[:,0:sz0002-2]   # -2 to be devided by 4
# ima_25=ima_2500[:,0:sz0002-2]  # -2 to be devided by 4
# sz001=GT25.shape[0]; # height
# sz002=GT25.shape[1]; # width
# print('sz001  ',sz001,'  sz002 ',sz002)

# sz01=int(sz001/2)
# sz02=int(sz002/2)
# print('sz01  ',sz01,'  sz02 ',sz02)
# GT50 = cv2.resize(GT25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
# ima_50 = cv2.resize(ima_25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)


## data2 LiDAR
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
GT000 =  np.double(np.load(cwd+'/SR_test_data/data2_LiDAR_25cm.npy'))
ima_2500 =  np.double(np.load(cwd+'/SR_test_data/data2_LiDAR_25cm.npy'))
name_file='data2_LiDAR'
sz0001=GT000.shape[0]; # height
sz0002=GT000.shape[1]; # width
sz00012=ima_2500.shape[0]; # height
sz00022=ima_2500.shape[1]; # width
pred_full=np.zeros((sz0001,sz0002))
imagt_full=np.zeros((sz0001,sz0002))
GT25=GT000[0:sz0001-8-2,0:sz0002-5-2] # pb of cooregistration a and impaire number and to be devided by 4
ima_25=ima_2500[0:sz0001-8-2,0:sz0002-5-2]   # pb of cooregistration a and impaire number and to be devided by 4

sz001=GT25.shape[0]; # height
sz002=GT25.shape[1]; # width
if(save_im==1):
    pred_full=np.zeros((sz0001,sz0002))
    imagt_full=np.zeros((sz0001,sz0002))

sz01=int(sz001/2)
sz02=int(sz002/2)
GT50 = cv2.resize(GT25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
ima_50 = cv2.resize(ima_25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)


## data2 photogrammetry
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GT000 =  np.double(np.load(cwd+'/SR_test_data/data2_LiDAR_25cm.npy'))
# ima_2500 =  np.double(np.load(cwd+'/SR_test_data/data2_photo_25cm.npy'))
# name_file='data2_photo'
# sz0001=GT000.shape[0]; # height
# sz0002=GT000.shape[1]; # width
# sz00012=ima_2500.shape[0]; # height
# sz00022=ima_2500.shape[1]; # width
# pred_full=np.zeros((sz0001,sz0002))
# imagt_full=np.zeros((sz0001,sz0002))
# GT25=GT000[0:sz0001-8-2,0:sz0002-5-2] # pb of cooregistration a and impaire number and to be devided by 4
# ima_25=ima_2500[7:sz00012-2,2:sz00022-1-2]  # pb of cooregistration a and impaire number and to be devided by 4

# sz001=GT25.shape[0]; # height
# sz002=GT25.shape[1]; # width
# if(save_im==1):
#     pred_full=np.zeros((sz0001,sz0002))
#     imagt_full=np.zeros((sz0001,sz0002))

# sz01=int(sz001/2)
# sz02=int(sz002/2)
# GT50 = cv2.resize(GT25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
# ima_50 = cv2.resize(ima_25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# data3 lidar
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# GT25 =  np.double(np.load(cwd+'/SR_test_data/data3_LiDAR_25cmDSM.npy'))
# ima_25 =  np.double(np.load(cwd+'/SR_test_data/data3_LiDAR_25cmDSM.npy'))
# name_file='data3_LiDAR'
# sz001=GT25.shape[0]; # height
# sz002=GT25.shape[1]; # width
# GT25[:,0]=GT25[:,1]
# GT25[:,sz002-1]=GT25[:,sz002-2]
# GT25[0,:]=GT25[1,:]
# GT25[sz001-1,:]=GT25[sz001-2,:]
# ima_25[:,0]=ima_25[:,1]
# ima_25[:,sz002-1]=ima_25[:,sz002-2]
# ima_25[0,:]=ima_25[1,:]
# ima_25[sz001-1,:]=ima_25[sz001-2,:]
# print('sz001  ',sz001,'  sz002 ',sz002)
# if(save_im==1):
#     pred_full=np.zeros((sz001,sz002))
#     imagt_full=np.zeros((sz001,sz002))


# sz01=int(sz001/2)
# sz02=int(sz002/2)
# print('sz01  ',sz01,'  sz02 ',sz02)
# GT50 = cv2.resize(GT25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
# ima_50 = cv2.resize(ima_25, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
# # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

sz21=int(sz01/2)
sz22=int(sz02/2)
print('sz21  ',sz21,'  sz22 ',sz22)
ima_Lin100 = cv2.resize(ima_50, dsize=(sz22, sz21), interpolation=cv2.INTER_LINEAR)
print('ima_Lin100 shape=',ima_Lin100.shape[0],'  ',ima_Lin100.shape[1])
ima_Lin100_50 = cv2.resize(ima_Lin100, dsize=(sz02, sz01), interpolation=cv2.INTER_LINEAR)
print('ima_Lin100_50 shape=',ima_Lin100_50.shape[0],'  ',ima_Lin100_50.shape[1])

if(full_image==0):
    sz022=np.int(sz02/2)
    sz222=np.int(sz22/2)
    
    sz02f=np.int(sz02/4)
else:
    sz022=0
    sz222=0
    sz02f=sz02

imagt=GT50 #[:,sz022:sz02] #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ima=ima_Lin100 #[:,sz222:sz22] #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ima_Lin100_50f=ima_Lin100_50[:,sz02-sz02f:sz02] # =++++++++++++++++++++++++++++++++++++++++++++++++++
# ima_Lin25f=ima_25[sz01-sz01f:sz01,:]
print('imagt shape=',imagt.shape[0],'  ',imagt.shape[1])
print('ima shape=',ima.shape[0],'  ',ima.shape[1])

print('ima_Lin100_50f shape=',ima_Lin100_50f.shape[0],'  ',ima_Lin100_50f.shape[1])


mn=ima.min()
mx=ima.max()

start=datetime.now()

sz1=ima.shape[0]
sz2=ima.shape[1]
sz1g=imagt.shape[0]
sz2g=imagt.shape[1]

    
pred_image50=np.zeros((sz1g,sz2g))
nb1=np.uint16(sz1/dim)
nb2=np.uint16(sz2/dim)
resid1=np.uint16(sz1%dim)
resid2=np.uint16(sz2%dim)
                     
resid1g=np.uint16(sz1g%dimG)
resid2g=np.uint16(sz2g%dimG)

for i1 in range(nb1):
    for i2 in range(nb2):
        ima_crop=ima[i1*dim:(i1+1)*dim,i2*dim:(i2+1)*dim]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
            
        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        # print('dimg  shape pred',dimG,'  ',predicted.shape)
        pred_image50[i1*dimG:(i1+1)*dimG,i2*dimG:(i2+1)*dimG]=predicted
            
if(resid1g>0):
    i1=nb1
    for i2 in range(nb2):
        ima_crop=ima[sz1-dim:sz1,i2*dim:(i2+1)*dim]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
          
        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image50[sz1g-dimG:sz1g,i2*dimG:(i2+1)*dimG]=predicted
            
if(resid2g>0):
    i2=nb2
    for i1 in range(nb1):
        ima_crop=ima[i1*dim:(i1+1)*dim,sz2-dim:sz2]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
            
        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image50[i1*dimG:(i1+1)*dimG,sz2g-dimG:sz2g]=predicted
        
if( (resid1g>0) & (resid2g>0) ):
    i1=nb1
    i2=nb2
    ima_crop=ima[sz1-dim:sz1,sz2-dim:sz2]
    ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
    
    predicted = model1.predict(ima_crop,verbose=0)
    predicted = np.reshape(predicted,(dimG,dimG))
    # predicted_im = (predicted*(mx-mn)) + mn
    pred_image50[sz1g-dimG:sz1g,sz2g-dimG:sz2g]=predicted

for i1 in range(nb1-1):
    for i2 in range(nb2):
        ima_crop=ima[i1*dim+int(dim/2):(i1+1)*dim+int(dim/2),i2*dim:(i2+1)*dim]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))

        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image50[i1*dimG+int(3*dimG/4):(i1+1)*dimG+int(dimG/4),i2*dimG:(i2+1)*dimG]=predicted[int(dimG/4):int(3*dimG/4),:]
 
for i1 in range(nb1):
    for i2 in range(nb2-1):
        ima_crop=ima[i1*dim:(i1+1)*dim,i2*dim+int(dim/2):(i2+1)*dim+int(dim/2)]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))

        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image50[i1*dimG:(i1+1)*dimG,i2*dimG+int(3*dimG/4):(i2+1)*dimG+int(dimG/4)]=predicted[:,int(dimG/4):int(3*dimG/4)]

for i1 in range(nb1-1):
    for i2 in range(nb2-1):
        ima_crop=ima[i1*dim+int(dim/2):(i1+1)*dim+int(dim/2),i2*dim+int(dim/2):(i2+1)*dim+int(dim/2)]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))

        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image50[i1*dimG+int(3*dimG/4):(i1+1)*dimG+int(dimG/4),i2*dimG+int(3*dimG/4):(i2+1)*dimG+int(dimG/4)]=predicted[int(dimG/4):int(3*dimG/4),int(dimG/4):int(3*dimG/4)]


print('pred_image50 min  ',pred_image50.min(),'  max ',pred_image50.max())
print('imalin min  ',mn,'  max ',mx)
print('imagt min  ',imagt.min(),'  max ',imagt.max())

pred_image50ts=pred_image50[:,sz2g-sz02f:sz2g] 
imagt=imagt[:,sz2g-sz02f:sz2g]
imaa=ima

print('imagt shape=',imagt.shape[0],'  ',imagt.shape[1])
print('pred_image50 shape=',pred_image50.shape[0],'  ',pred_image50.shape[1])


## second part from 50 to 25
# *********************************************************************
# *******************************************************
# ******************************************************************
dim=128  
dimG=256
model1=load_model(cwd +'/SRUNet_50_25.h5', custom_objects={'custom_loss': custom_loss})

ima_Lin100_50_25 = cv2.resize(ima_Lin100_50, dsize=(sz002, sz001), interpolation=cv2.INTER_LINEAR)
print('ima_Lin100_50_25 shape=',ima_Lin100_50_25.shape[0],'  ',ima_Lin100_50_25.shape[1])


if(full_image==0):
    sz012=np.int(sz002/2)
    sz212=np.int(sz02/2)
    
    sz02f=np.int(sz002/4)
else:
    sz012=0
    sz212=0
    sz02f=sz002


imagt=GT25 #[:,sz012:sz002] #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ima=pred_image50 ##[:,sz212:sz02] #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('*********************-----------------------------------------.........................')
print('imagt shape=',imagt.shape[0],'  ',imagt.shape[1])
print('pred_image50 shape=',pred_image50.shape[0],'  ',pred_image50.shape[1])
print('sz212 sz02=',sz212,'  ',sz02)
print('imagt shape=',imagt.shape[0],'  ',imagt.shape[1])

ima_Lin100_50_25f=ima_Lin100_50_25[:,sz002-sz02f:sz002] # =++++++++++++++++++++++++++++++++++++++++++++++++++
# ima_Lin100_50_25f=ima_Lin100_50_25[sz001-sz01f:sz001,:] # =++++++++++++++++++++++++++++++++++++++++++++++++++

# ima_Lin25f=ima_25[sz01-sz01f:sz01,:]
print('imagt shape=',imagt.shape[0],'  ',imagt.shape[1])
print('ima shape=',ima.shape[0],'  ',ima.shape[1])

print('ima_Lin100_50_25f shape=',ima_Lin100_50_25f.shape[0],'  ',ima_Lin100_50_25f.shape[1])


mn=ima.min()
mx=ima.max()

start=datetime.now()

sz1=ima.shape[0]
sz2=ima.shape[1]
sz1g=imagt.shape[0]
sz2g=imagt.shape[1]
    
    
pred_image25=np.zeros((sz1g,sz2g))
nb1=np.uint16(sz1/dim)
nb2=np.uint16(sz2/dim)
resid1=np.uint16(sz1%dim)
resid2=np.uint16(sz2%dim)
                     
resid1g=np.uint16(sz1g%dimG)
resid2g=np.uint16(sz2g%dimG)

for i1 in range(nb1):
    for i2 in range(nb2):
        ima_crop=ima[i1*dim:(i1+1)*dim,i2*dim:(i2+1)*dim]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
            
        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image25[i1*dimG:(i1+1)*dimG,i2*dimG:(i2+1)*dimG]=predicted
            
if(resid1g>0):
    i1=nb1
    for i2 in range(nb2):
        ima_crop=ima[sz1-dim:sz1,i2*dim:(i2+1)*dim]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
          
        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image25[sz1g-dimG:sz1g,i2*dimG:(i2+1)*dimG]=predicted
            
if(resid2g>0):
    i2=nb2
    for i1 in range(nb1):
        ima_crop=ima[i1*dim:(i1+1)*dim,sz2-dim:sz2]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
            
        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image25[i1*dimG:(i1+1)*dimG,sz2g-dimG:sz2g]=predicted
        
if( (resid1g>0) & (resid2g>0) ):
    i1=nb1
    i2=nb2
    ima_crop=ima[sz1-dim:sz1,sz2-dim:sz2]
    ima_crop=np.reshape(ima_crop,(1,dim,dim,1))
    
    predicted = model1.predict(ima_crop,verbose=0)
    predicted = np.reshape(predicted,(dimG,dimG))
    # predicted_im = (predicted*(mx-mn)) + mn
    pred_image25[sz1g-dimG:sz1g,sz2g-dimG:sz2g]=predicted

for i1 in range(nb1-1):
    for i2 in range(nb2):
        ima_crop=ima[i1*dim+int(dim/2):(i1+1)*dim+int(dim/2),i2*dim:(i2+1)*dim]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))

        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image25[i1*dimG+int(3*dimG/4):(i1+1)*dimG+int(dimG/4),i2*dimG:(i2+1)*dimG]=predicted[int(dimG/4):int(3*dimG/4),:]
 
for i1 in range(nb1):
    for i2 in range(nb2-1):
        ima_crop=ima[i1*dim:(i1+1)*dim,i2*dim+int(dim/2):(i2+1)*dim+int(dim/2)]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))

        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image25[i1*dimG:(i1+1)*dimG,i2*dimG+int(3*dimG/4):(i2+1)*dimG+int(dimG/4)]=predicted[:,int(dimG/4):int(3*dimG/4)]

for i1 in range(nb1-1):
    for i2 in range(nb2-1):
        ima_crop=ima[i1*dim+int(dim/2):(i1+1)*dim+int(dim/2),i2*dim+int(dim/2):(i2+1)*dim+int(dim/2)]
        ima_crop=np.reshape(ima_crop,(1,dim,dim,1))

        predicted = model1.predict(ima_crop,verbose=0)
        predicted = np.reshape(predicted,(dimG,dimG))
        # predicted_im = (predicted*(mx-mn)) + mn
        pred_image25[i1*dimG+int(3*dimG/4):(i1+1)*dimG+int(dimG/4),i2*dimG+int(3*dimG/4):(i2+1)*dimG+int(dimG/4)]=predicted[int(dimG/4):int(3*dimG/4),int(dimG/4):int(3*dimG/4)]


print('pred_image25 min  ',pred_image25.min(),'  max ',pred_image25.max())
print('imalin min  ',mn,'  max ',mx)
print('imagt min  ',imagt.min(),'  max ',imagt.max())

if(save_im==1):
    print('*********************************  save **************************************')
    if(full_image==1):
        pred_image2=pred_image25
        ima_Lin50_252=ima_Lin100_50_25
        imagt2=imagt
    else:
        pred_image2[:,sz002-sz02f:sz002]=pred_image25[:,sz002-sz02f:sz002]
        ima_Lin50_252[:,sz002-sz02f:sz002]=ima_Lin100_50_25[:,sz002-sz02f:sz002]
        imagt2[:,sz002-sz02f:sz002]=imagt[:,sz002-sz02f:sz002]
    
    np.save(cwd +'/prediction/'+name_file+'_res.npy',pred_image2)
    np.save(cwd +'/prediction/'+name_file+'_GT.npy',imagt2)
    
    
pred_image25ts=pred_image25[:,sz002-sz02f:sz002] # ++++++++++++++++++++++++++++++++++++++++++++++++++++

imagt=imagt[:,sz002-sz02f:sz002]
# pred_image25ts=pred_image25[sz1g-sz01f:sz1g,:] # ++++++++++++++++++++++++++++++++++++++++++++++++++++
# imagt=imagt[sz1g-sz01f:sz1g,:]
print('imagt shape=',imagt.shape[0],'  ',imagt.shape[1])
print('pred_image25 shape=',pred_image25.shape[0],'  ',pred_image25.shape[1])
print('*******************************************')
print('before mn mx---')
# ima_Lin25f=ima_Lin25[sz01f:sz1,:]
MSE = mse(pred_image25ts,imagt,bands)
RMSE = rmse(pred_image25ts,imagt,bands) # mae(imagt255,predicted255,bands)
# PSNR = tf.image.psnr(, predicted255 , max_val=255)
PSNR=psnr(pred_image25ts,imagt)
SSIM = ssim(pred_image25ts,imagt, multichannel=False)

print('MSE = %.4f' % MSE)
print('eRMSE = %.4f' % RMSE)
print('PSNR = %.3f' % PSNR)
SSIM100=SSIM*100
print('SSIM = %.3f' % SSIM100)
