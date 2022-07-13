#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:21:21 2019


"""

#%%   
import glob
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

#%% Utility Functions


def percentile_norm(datas):
    
    std1  = np.std(datas)
    mean  = np.mean(datas)
    datas -=  mean
    datas  /= std1
    
    return datas

def generate_batch_norm(batch):
    data = []
    
    for img in batch:
        #print('img = '+str(img))
        img_data = sitk.ReadImage(img) 
                
        img_data = sitk.GetArrayFromImage(img_data)
        
        img_data = img_data.astype('float32')
        img_data = percentile_norm(img_data)
        data.append(img_data)

    data = np.stack(data)


    data = np.reshape(data, (data.shape[0],data.shape[2],data.shape[1],1))
    return data


#%%
""" 
  Loading the model with the trained weights
"""


from hybrid_model2 import  wnet_dc_nonorm2


#model = create_unet()# Create the U-Net
model = wnet_dc_nonorm2()

model.load_weights('weights.60_.hdf5')


#%% Generate Nifti prediction on test set images


z_free_breath = os.listdir("test_images")

img_path = list()

for sli in z_free_breath:
    img_path.append("test_images\\"+sli)
    
    


for i in  range(np.size(img_path)):   
    im_arra = generate_batch_norm([img_path[i]])
   
 
   
    im_pred = model.predict(im_arra)
   
   
    im_pred = im_pred[0,:,:,0]
    im_arra = im_arra[0,:,:,0]
    
    
    plt.figure()
    plt.imshow()
   







