#!/home/dongyi/anaconda3/envs/paddle_env/bin/python3
# -*- coding: utf-8 -*-
"""
@author: marco
"""

# Configuration 
### import necessary packages
# import sys 
# sys.path.append('/home/aistudio/external-libraries')
#import os
#import cv2
#import random
import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances 
#from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix
#import matplotlib.pylab as plt

import paddle
import paddle.nn as nn
#from paddle.vision.models import resnet50, resnet101, resnet152
#from paddle.io import Dataset
#import transforms as trans 
#import functional as F
#import logging 
#from datetime import datetime

# For coordinates without confidence
# compute ED and loss of multiple points
# cal_ed computes the average ED of a batch when training
def cal_ed(logit, label):
    ed_loss = []
    for i in range(logit.shape[0]):  
        for j in range(logit.shape[1]):  # logit.shape[1]==5
            if label[i][j].max() >= 0.9:
                kx_l= np.where(label[i][j]==label[i][j].max())[1][0]
                ky_l= np.where(label[i][j]==label[i][j].max())[0][0]

                kx_p= np.where(logit[i][j]==logit[i][j].max())[1][0]  # predicted keypoints
                ky_p= np.where(logit[i][j]==logit[i][j].max())[0][0]

                ed_tmp = euclidean_distances([kx_p, ky_p],[kx_l, ky_l])
                ed_loss.append(ed_tmp)

    ed_l = sum(ed_loss)/len(ed_loss) 
    return ed_l

# cal_ed_val computes the average ED of a batch when validation
def cal_ed_val(logit, label):
    ed_loss = []
    h= logit.shape[2]
    w= logit.shape[3]
    # print(h,w)
    for i in range(logit.shape[0]):  
        for j in range(logit.shape[1]):  # logit.shape[1]==5
            if label[i][j].max() >= 0.9:
                kx_l= np.where(label[i][j]==label[i][j].max())[1][0]
                ky_l= np.where(label[i][j]==label[i][j].max())[0][0]

                kx_p= np.where(logit[i][j]==logit[i][j].max())[1][0]  # predicted keypoints
                ky_p= np.where(logit[i][j]==logit[i][j].max())[0][0]
                
                # calculate euclidean_distances by normalizad kx, ky
                ed_tmp = euclidean_distances([[kx_p/w, ky_p/h]],[[kx_l/w, ky_l/h]]) # 2D arrays are expected 
                # calculate euclidean_distances by absolute kx, ky
                # ed_tmp = euclidean_distances([[kx_p, ky_p]],[[kx_l, ky_l]])

                ed_loss.append(ed_tmp)
    
    ed_l = sum(ed_loss)/len(ed_loss)
    
    return ed_l

# loss = alpha*MSE Loss + (1-alpha) *ED
def cal_loss(logit, label, alpha = 0.618):  #  0.75 0.618
    """
    logit: shape [batch, ndim]
    label: shape [batch, ndim]
    ndim = 2 represents coordinate_x and coordinaate_y
    alpha: weight for MSELoss and 1-alpha for ED loss
    return: combine MSELoss and ED Loss for x and y, shape [batch, 1]
    """
    h= logit.shape[2]
    w= logit.shape[3]
    # MSE Loss, a.k.a L2 Loss
    mse_loss = nn.MSELoss(reduction='mean')
    # As heatmap values are in (0,1), L1 Loss is more sensitive than L2 Loss, L2 Loss is more sensitive than SmoothL1Loss
    # mse_loss = nn.L1Loss(reduction='mean')

    mse_coordinates = []
    ed_loss = []
    for i in range(logit.shape[0]):  
        for j in range(logit.shape[1]):  # logit.shape[1]==5
            mse_coordinates.append(mse_loss(logit[i][j],label[i][j]))  # MSE loss or L1 Loss for every pixels in the heatmap

            if label[i][j].max() >= 0.9:  # euclidean_distances loss for keypoints
                kx_l= np.where(label[i][j]==label[i][j].max())[1][0]
                ky_l= np.where(label[i][j]==label[i][j].max())[0][0]

                kx_p= np.where(logit[i][j]==logit[i][j].max())[1][0]  # predicted keypoints
                ky_p= np.where(logit[i][j]==logit[i][j].max())[0][0]
                
                # calculate euclidean_distances by normalizad kx, ky
                # ed_tmp = euclidean_distances([[kx_p/w, ky_p/h]],[[kx_l/w, ky_l/h]]) # 2D arrays are expected 
                _p = paddle.to_tensor([kx_p/w, ky_p/h],stop_gradient=False)
                _l = paddle.to_tensor([kx_l/w, ky_l/h],stop_gradient=False)
                ed_tmp = mse_loss(_p,_l)

                ed_loss.append(ed_tmp)
    mse_l = sum(mse_coordinates)/len(mse_coordinates)
    ed_l = sum(ed_loss)/len(ed_loss)

    loss = alpha * mse_l + (1-alpha) * ed_l
    # print(logit)
    # print(label)
    # print('ed_l', ed_l)
    # print('mse_l', mse_l)
    # print('alpha', alpha)
    # print('loss in function', loss)
    return loss

