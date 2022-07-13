#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:26:41 2020

@author: jose
"""

import keras 
from keras import Input
from keras.models import Model
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import Concatenate, Add, Lambda, Conv2D, Lambda, MaxPooling2D, concatenate, UpSampling2D, BatchNormalization 
from keras.preprocessing.image import array_to_img
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    K.print_tensor(intersection, message="Dice intersection:")
    return -((2. * intersection + K.epsilon()) / (K.sum(y_true_f)
                                                  + K.sum(y_pred_f)
                                                  + K.epsilon()))
    

def nrmse(y_true, y_pred):
    denom = K.sqrt(K.mean(K.square(y_true), axis=(1,2,3)))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=(1,2,3)))\
    /denom
    
def nrmse_loss(y_true,y_pred):
   return nrmse(y_true, y_pred)
    
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)    
 

   
   
def ifft_layer(kspace):
    real = Lambda(lambda kspace : kspace[:,:,:,0])(kspace)
    imag = Lambda(lambda kspace : kspace[:,:,:,1])(kspace)
    kspace_complex = tf.dtypes.complex(real,imag)
    rec1 = tf.keras.backend.abs(tf.signal.ifft(kspace_complex))
    rec1 = tf.expand_dims(rec1, -1)
    return rec1   
 
def wnet_dc(mu1,sigma1,mu2,sigma2,mask,H=256,W=256,channels = 2,kshape = (3,3),kshape2=(3,3)):
    inputs = Input(shape=(H,W,channels))
    neu   = 40
    
    conv1 = Conv2D(neu, kshape, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(neu, kshape, activation='relu', padding='same')(conv1)
    conv1 = Conv2D(neu, kshape, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(neu*2, kshape, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(neu*2, kshape, activation='relu', padding='same')(conv2)
    conv2 = Conv2D(neu*2, kshape, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(neu*3, kshape, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(neu*3, kshape, activation='relu', padding='same')(conv3)
    conv3 = Conv2D(neu*3, kshape, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(neu*4, kshape, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(neu*4, kshape, activation='relu', padding='same')(conv4)
    conv4 = Conv2D(neu*4, kshape, activation='relu', padding='same')(conv4)
    
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3],axis=-1)
    conv5 = Conv2D(neu*3, kshape, activation='relu', padding='same')(up1)
    conv5 = Conv2D(neu*3, kshape, activation='relu', padding='same')(conv5)
    conv5 = Conv2D(neu*3, kshape, activation='relu', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2],axis=-1)
    conv6 = Conv2D(neu*2, kshape, activation='relu', padding='same')(up2)
    conv6 = Conv2D(neu*2, kshape, activation='relu', padding='same')(conv6)
    conv6 = Conv2D(neu*2, kshape, activation='relu', padding='same')(conv6)
    
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1],axis=-1)
    conv7 = Conv2D(neu, kshape, activation='relu', padding='same')(up3)
    conv7 = Conv2D(neu, kshape, activation='relu', padding='same')(conv7)
    conv7 = Conv2D(neu, kshape, activation='relu', padding='same')(conv7)
    
    conv8 = Conv2D(2, (1, 1), activation='linear')(conv7)
    conv8_dc = Lambda(lambda conv8 : conv8*mask)(conv8)
     
    
    res1 = Add()([conv8_dc,inputs])
    res1_scaled = Lambda(lambda res1 : (res1*sigma1+mu1))(res1)
    
    rec1 = Lambda(ifft_layer)(res1_scaled)
    rec1_norm = Lambda(lambda rec1 : (rec1-mu2)/sigma2)(rec1)
    
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(rec1_norm)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(pool4)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(pool5)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(pool6)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(up4)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(up5)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(up6)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    
    out = Conv2D(1, (1, 1), activation='linear')(conv15)
    model = Model(inputs=inputs, outputs=[res1_scaled,out])
    
    model.compile(optimizer=Adam(lr=1e-3,decay = 1e-7),
                 loss=[nrmse,nrmse], metrics=[nrmse_loss], loss_weights=[0.01, 0.99] )
    return model
 
#==============================================================================
# def wnet_dc_nonorm2(H=256,W=256,channels = 2,kshape = (3,3),kshape2=(3,3)):
# 
#     inputs = Input((256, 256,1))
#  
#    
#     conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(inputs)
#     conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
#     conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
#     
#     conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(pool4)
#     conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
#     conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
#     pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)
#     
#     conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(pool5)
#     conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
#     conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
#     pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)
#     
#     conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(pool6)
#     conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
#     conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
#     
#     up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
#     conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(up4)
#     conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
#     conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
#     
#     up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
#     conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(up5)
#     conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
#     conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
#     
#     up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
#     conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(up6)
#     conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
#     conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
#     
#     out = Conv2D(1, (1, 1), activation='linear')(conv15)
#     model = Model(inputs=inputs, outputs=[out])
#     
#     model.compile(optimizer=Adam(lr=1e-3,decay = 1e-7),
#==============================================================================
#                 loss=[nrmse], metrics=[nrmse_loss])
    
#    return model 
def wnet_dc_nonorm2(H=256,W=256,channels = 2,kshape = (3,3),kshape2=(3,3)):

    inputs = Input((256, 256,1))
 
   
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(inputs)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    conv9 = Conv2D(48, kshape2, activation='relu', padding='same')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(pool4)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    conv10 = Conv2D(64, kshape2, activation='relu', padding='same')(conv10)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(pool5)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    conv11 = Conv2D(128, kshape2, activation='relu', padding='same')(conv11)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv11)
    
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(pool6)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    conv12 = Conv2D(256, kshape2, activation='relu', padding='same')(conv12)
    
    up4 = concatenate([UpSampling2D(size=(2, 2))(conv12), conv11],axis=-1)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(up4)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    conv13 = Conv2D(128, kshape2, activation='relu', padding='same')(conv13)
    
    up5 = concatenate([UpSampling2D(size=(2, 2))(conv13), conv10],axis=-1)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(up5)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    conv14 = Conv2D(64, kshape2, activation='relu', padding='same')(conv14)
    
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv14), conv9],axis=-1)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(up6)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    conv15 = Conv2D(48, kshape2, activation='relu', padding='same')(conv15)
    
    out = Conv2D(1, (1, 1), activation='linear')(conv15)
    model = Model(inputs=inputs, outputs=[out])
    
    model.compile(optimizer=Adam(lr=1e-3,decay = 1e-7),
                 loss=[nrmse], metrics=[nrmse_loss])
    
    return model   
   
   
def create_unet2():
    '''
    Creates a U-Net
    '''
    print('Creating U-Net...')

    # First, we have to provide the dimensions of the input images
    inputs = Input((256, 256,1))

    
    
#######Image_space Network ########################3
    
    conv13 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs )
    conv13 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv13)
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv13)
    print('conv13 shape:', conv13.shape)
    print('pool6 shape:', pool6.shape)

    conv14 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool6)
    conv14 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv14)
    pool7 = MaxPooling2D(pool_size=(2, 2))(conv14)
    print('conv14 shape:', conv14.shape)
    print('pool7 shape:', pool7.shape)

    conv15 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool7)
    conv15 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv15)
    pool8 = MaxPooling2D(pool_size=(2, 2))(conv15)
    print('conv15 shape:', conv15.shape)
    print('pool8 shape:', pool8.shape)

    conv16 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool8)
    conv16 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv16)
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv16)

    print('conv16 shape:', conv16.shape)
    print('pool9 shape:', pool9.shape)
    
    
    conv17 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool9)
    conv17 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv17)
    pool10 = MaxPooling2D(pool_size=(2, 2))(conv17)
    print('conv17 shape:', conv17.shape)
    print('pool10 shape:', pool10.shape)      
    
    conv18 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool10)
    conv18 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv18)
    drop7 = Dropout(0.5)(conv18)
    
    print('conv18 shape:', conv18.shape) 
    
    
    up11 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(drop7))  # Changed
    merge7 = Concatenate(axis=3)([conv17, up11])
    conv19 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv19 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv19)
    print('conv19 shape:', conv19.shape)     
    
    
    
    up12 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv7))  # Changed
    merge8 = Concatenate(axis=3)([conv16, up12])
    conv20 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv20 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv20)
    print('conv20 shape:', conv20.shape)    


    up13 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv20))  # Changed
    merge9 = Concatenate(axis=3)([conv15, up13])
    conv21 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv21 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv21)
    print('conv21 shape:', conv21.shape)

    up14 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv21))
    merge10 = Concatenate(axis=3)([conv14, up14])
    conv22 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge10)
    conv22 = Conv2D(32, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv22)
    print('conv22 shape:', conv22.shape)

    up15 = Conv2D(32, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv22))
    merge11 = Concatenate(axis=3)([conv13, up15])
    conv23 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge11)
    conv23 = Conv2D(16, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv23)
    conv23 = Conv2D(4, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv23)
    print('conv23 shape:', conv23.shape)




    conv24 = Conv2D(1, 1, activation='linear')(conv23)   


    print('conv12 shape:', conv12.shape)





    model = Model(inputs=inputs, outputs=[conv24])
 #   model = Model(inputs=inputs, outputs=conv24)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=[nrmse], metrics=[nrmse_loss])
    
    

#    model.compile(optimizer=Adam(lr=1e-4),
#                  loss=nrmse, metrics=[nrmse_loss])    

#    model.load_weights('/media/data/Prostate_data_sets/data_dory/LungImagingMarching/OneDrive-2020-10-09/weights/weights.5000_.hdf5')  # Load the pre-trained U-Net



    print('Got U-Net!')

    return model
   
