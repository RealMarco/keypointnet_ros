#!/home/dongyi/anaconda3/envs/paddle_env/bin/python3
# -*- coding: utf-8 -*-
"""
@author: marco
"""
### import necessary packages
# import sys 
# sys.path.append('/home/aistudio/external-libraries')
#import os
import cv2
#import random
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics.pairwise import euclidean_distances 
#from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
#import matplotlib.pylab as plt

#import paddle
#import paddle.nn as nn
#from paddle.vision.models import resnet50, resnet101, resnet152
from paddle.io import Dataset
#import transforms as trans 
#import functional as F
#import logging 
#from datetime import datetime
from inference.config import image_size
### Class Dataset of the first network
### load the img images from the data folder, 
### and extract the corresponding ground truth to generate training samples
def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma=64):  # faster, sigma controls the size of gaussian kernel
    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)
    X = X - c_x
    Y = Y - c_y
    D2 = X * X + Y * Y
    E2 = 2.0 * sigma * sigma
    Exponent = D2 / E2
    heatmap = np.exp(-Exponent).astype('float32')
    return heatmap

def SegmentedHeatmap(img,heatmap1):
    img=img.transpose(1, 2, 0) # C,H,W to H,W,C
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # find the corner pixel value in HSV
    r, g, b = cv2.split(img)
    b1, g1, r1 = int(b[0,0]),int(g[0,0]),int(r[0,0])
    b2, g2, r2 = int(b[0,-1]),int(g[0,-1]),int(r[0,-1])
    b3, g3, r3 = int(b[-1,0]),int(g[-1,0]),int(r[-1,0])
    b4, g4, r4 = int(b[-1,-1]),int(g[-1,-1]),int(r[-1,-1])
    (bg_h, bg_s, bg_v) = (round((r1+r2+r3+r4)/4),round((g1+g2+g3+g4)/4),round((b1+b2+b3+b4)/4))       
    #fill = (img[0,0] + img[0,-1] + img[-1,0] + img[-1,-1])/4  #ERROR the dtype of img[i,j] is unit8, so the sum of img[i,j]s is in range 0-255， must convert it to int (long int)
    #(bg_h, bg_s, bg_v) = (round(fill[0]),round(fill[1]),round(fill[2]))
    mask = cv2.inRange(img, (bg_h, bg_s, bg_v), (bg_h, bg_s, bg_v))
    
    # find the most frequent pixel value in HSV
    img_temp = img.copy() # deep copy, since img_temp = img is shallow copy 
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    # img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]
    # print(unique[np.argmax(counts)])
    bg_h2, bg_s2, bg_v2 = int(unique[np.argmax(counts)][0]), int(unique[np.argmax(counts)][1]), int(unique[np.argmax(counts)][2])
    mask2 = cv2.inRange(img, (bg_h2, bg_s2, bg_v2), (bg_h2, bg_s2, bg_v2))

    mask_all = cv2.bitwise_or(mask, mask2)
    # To erode the inside highlight edge
    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    #mask_all = cv2.dilate(mask_all, kernel) 
    mask_all= cv2.medianBlur(mask_all, 5)

    mask_inv = cv2.bitwise_not(mask_all)
    seg_heatmap = cv2.bitwise_and(heatmap1, heatmap1, mask=mask_inv)

    return seg_heatmap

def SegmentedHeatmap_nonkey(img):
    img=img.transpose(1, 2, 0) # 3,512,512 C,H,W to H,W,C
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # find the corner pixel value in HSV
    h, s, v = cv2.split(img)  
    v1, s1, h1 = int(v[0,0]),int(s[0,0]),int(h[0,0])
    v2, s2, h2 = int(v[0,-1]),int(s[0,-1]),int(h[0,-1])
    v3, s3, h3 = int(v[-1,0]),int(s[-1,0]),int(h[-1,0])
    v4, s4, h4 = int(v[-1,-1]),int(s[-1,-1]),int(h[-1,-1])
    (bg_h, bg_s, bg_v) = (round((h1+h2+h3+h4)/4),round((s1+s2+s3+s4)/4),round((v1+v2+v3+v4)/4))       
    mask = cv2.inRange(img, (bg_h, bg_s, bg_v), (bg_h, bg_s, bg_v))
    
    # find the most frequent pixel value in HSV
    img_temp = img.copy() # deep copy, since img_temp = img is shallow copy 
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    bg_h2, bg_s2, bg_v2 = int(unique[np.argmax(counts)][0]), int(unique[np.argmax(counts)][1]), int(unique[np.argmax(counts)][2])
    mask2 = cv2.inRange(img, (bg_h2, bg_s2, bg_v2), (bg_h2, bg_s2, bg_v2))

    mask_all = cv2.bitwise_or(mask, mask2)

    mask_all= cv2.medianBlur(mask_all, 5)

    mask_inv = cv2.bitwise_not(mask_all)
    # Method 1
    # seg_heatmap = mask_inv * 0.1 # error: OpenCV(4.1.1) /io/opencv/modules/core/src/merge.dispatch.cpp:129: 
                                 # error: (-215:Assertion failed) mv[i].size == mv[0].size && mv[i].depth() == depth in function 'merge'
    # seg_heatmap = (mask_inv * 0.0004).astype('float32')
    # Method 2
    heatmap1= (np.ones((img.shape[0], img.shape[1]))*0.4).astype('float32') #*0.04 # H, W
    seg_heatmap = cv2.bitwise_and(heatmap1, heatmap1, mask=mask_inv)

    return seg_heatmap

class imgDataset(Dataset):
    def __init__(self,  image_file, label_file=None, img_transforms=None,filelists=None,  mode='train', index_mode = False):
        super(imgDataset, self).__init__()
        self.img_transforms =  img_transforms
        self.mode = mode.lower()
        self.image_path = image_file
        # image_idxs = os.listdir(self.image_path)
        image_idxs = self.image_path   # the list of img images' path
        self.label_file = label_file
        self.index_mode = index_mode
        # self.crop=crop

        if self.mode == 'train':
            label = {row['img_name']: row[1:].values 
                        for _, row in pd.read_excel(label_file).iterrows()}
            self.file_list = [[image_idxs[i], label[image_idxs[i].split('/')[-1]]] for i in range(len(image_idxs))] # split('/')[-1][:-4]
            # print(self.file_list)

        elif self.mode == 'test':
            self.file_list = [[image_idxs[i], None] for i in range(len(image_idxs))]
        
        elif self.mode == 'deploy':
            self.file_list = [[image_idxs[i], None] for i in range(len(image_idxs))] # image_idxs[i] is an image when self.mode == "deploy"
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists] 
   
    def __getitem__(self, idx):
        if self.mode == "deploy":
            img0, label = self.file_list[idx] # image_idxs[i] is an image when self.mode == "deploy"
            img = img0[:, :, ::-1] # BGR -> RGB
        else: # self.mode == "train" or "test"
            real_index, label = self.file_list[idx]
            img_path = real_index    # real_index = absolute path of imgs 
            img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB        
        
        h,w,c = img.shape   # h,w,c is the shape
        fh, fw = h,w   # fh, fw is the shape of original images acquired from object detection.

        if self.mode == 'train':
            confidence= (float(label[0]), float(label[3]), float(label[6]), float(label[9]) , float(label[12]))
            x         = (float(label[1]), float(label[4]), float(label[7]), float(label[10]), float(label[13]))  # 
            y         = (float(label[2]), float(label[5]), float(label[8]), float(label[11]), float(label[14]))
            fx = x
            fy = y

            # Normalization
            label_nor = (tuple(fi/fw for fi in fx), tuple(fj/fh for fj in fy))  
            label_nor = np.array(label_nor).astype('float32').reshape(2,5)  # from tuple to np.array
            confidence= np.array(confidence).astype('float32').reshape(5)
            
            # Data Augmengtation
            if self.img_transforms is not None:
                img_re, label_nor[0], label_nor[1]= self.img_transforms(img, label_nor[0], label_nor[1],confidence)
                img_re = cv2.resize(img_re,(image_size, image_size)) # ignore it when we use RandomResizedCrop
            
            img = img_re.transpose(2, 0, 1) # H, W, C -> C, H, W
            # img = img_re.astype(np.float32)

            label_nor = label_nor.reshape(10, order ='F') # reshape to a 1x10 ndarray [toe_end_x toe_end_y heel...] 
            # label_nor =  np.append(label_nor, confidence)  
            
            c,h,w = img.shape   # h, w is the shape of current images acquired from object detection.

            if confidence[0] == 1.0:
                label_heatmaps = CenterLabelHeatMap(w, h, w*label_nor[2*0], h*label_nor[2*0+1], 32) # heatmap method
                # label_heatmaps = SegmentedHeatmap(img, label_heatmaps)
            else:
                label_heatmaps = CenterLabelHeatMap(w, h, w*label_nor[2*0], h*label_nor[2*0+1], 0) # heatmap method
                # label_heatmaps = SegmentedHeatmap_nonkey(img) # non-keypoint heatmap is low-value segmentation map
            # print(label_heatmaps.shape)
            for j in range(1, len(label_nor)//2):
                if confidence[j] == 1.0:
                    heatmap1 = CenterLabelHeatMap(w, h, w*label_nor[2*j], h*label_nor[2*j+1], 32)
                    # heatmap1 = SegmentedHeatmap(img, heatmap1)
                else:
                    heatmap1 = CenterLabelHeatMap(w, h, w*label_nor[2*j], h*label_nor[2*j+1], 0)
                    # heatmap1 = SegmentedHeatmap_nonkey(img)  # non-keypoint heatmap is low-value segmentation map
                label_heatmaps = cv2.merge([label_heatmaps, heatmap1])
            label_heatmaps = label_heatmaps.transpose(2, 0, 1) # H, W, C -> C, H, W
            # print(type(label_heatmaps))
            # print(label_heatmaps.shape)
            if self.index_mode == False:
                return img, label_heatmaps # return img, label_nor
            else: # self.index_mode == True
                return img, label_heatmaps, real_index
        
        elif self.mode == 'test':
            # Data Augmengtation
            if self.img_transforms is not None:
                img_re = self.img_transforms(img)
                img_re = cv2.resize(img_re,(image_size, image_size)) # ignore it when we use RandomResizedCrop
            img = img_re.transpose(2, 0, 1) # H, W, C -> C, H, W
            c,h,w = img.shape   # h, w is the shape of current images acquired from object detection.
            return img, real_index, h, w, fh, fw  #  fh, fw is the shape of original images, while h, w is the shape of current images
        
        elif self.mode == 'deploy':
            # Data Augmengtation
            if self.img_transforms is not None:
                img_re = self.img_transforms(img)
                img_re = cv2.resize(img_re,(image_size, image_size)) # ignore it when we use RandomResizedCrop
            img = img_re.transpose(2, 0, 1) # H, W, C -> C, H, W
            c,h,w = img.shape   # h, w is the shape of current images acquired from object detection.
            return img, idx, h, w, fh, fw  #  fh, fw is the shape of original images, while h, w is the shape of current images
        else:
            print('No such mode')
            return None

    def __len__(self):
        return len(self.file_list)
