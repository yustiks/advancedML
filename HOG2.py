# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 14:05:47 2018

@author: aswin
"""
from skimage import feature
from skimage import exposure
from skimage.feature import hog
from skimage import data
import skimage
import cv2
import imutils
import os
from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np
from skimage import color
from skimage import io
import scipy

def read_images():
    images = []
    images_gx = []
    images_gy = []
    laplacian_Out = []
    magnitude = []
    out_image = []

    for image_name in os.listdir(r'by_style\training'):
        impath = os.path.join(r'by_style\training', image_name)
        print(impath)
        img = cv2.imread(impath,0)
        img = img.astype('float64')
        img = np.float64(img) / 255.0
        laplacian = cv2.Laplacian(img,cv2.CV_64F)
        gx = cv2.Sobel(laplacian, cv2.CV_64F, 1, 0, ksize=1)
        absGx = np.absolute(gx)
        gy = cv2.Sobel(laplacian, cv2.CV_64F, 0, 1, ksize=1)
        absGy = np.absolute(gy)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            
        fd, hog_image = hog(absGx, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualise=True)
        out_image.append(hog_image)
        images.append(np.array(img))    
        images_gx.append(absGx)
        images_gy.append(absGy)
        laplacian_Out.append(laplacian)
        magnitude.append(mag)
    return images, images_gx, images_gy, laplacian_Out, magnitude, out_image 

#==============================================================================
# def HOG_Descriptor():
#     out_image = []
#     for image_name in os.listdir(r'by_style\training'):
#         impath = os.path.join(r'by_style\training', image_name)
#         print(impath)
#         img = cv2.imread(impath,0)
#         img = img.astype('float64')
#         img = np.float64(img) / 255.0
#     
#         fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
#                 cells_per_block=(1, 1), visualise=True)
#         out_image.append(hog_image)
# 
# 
# #    hist = hog.compute(image)
#     print("OK")
# #    hog = cv2.HOGDescriptor()
#     return out_image
#==============================================================================

if __name__ == "__main__":  
    pic = 278
    data,X ,Y, Laplac, Mag, HOG = read_images()
    print('Data shape:', np.array(data).shape)
    print('Data Type:', np.array(data).dtype)


    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5)
    ax1.imshow(data[pic])
    ax2.imshow(X[pic])
    ax3.imshow(Y[pic])
    ax4.imshow(Mag[pic])
    ax5.imshow(HOG[pic])
#==============================================================================
#     hog_val = HOG_Descriptor()
#     plt.imshow(hog_val[21])
#==============================================================================
