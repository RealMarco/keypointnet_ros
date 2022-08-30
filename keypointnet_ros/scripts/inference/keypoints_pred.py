#!/home/dongyi/anaconda3/envs/paddle_env/bin/python
# -*- coding: utf-8 -*-
"""
@author: marco
"""

### import necessary packages
### system modules
#import sys 
#from sys import path
#path.append(0, sys.path[0] + '\\utils')
#sys.path.append('..')
#sys.path.append('/home/dongyi/ur_ws/src/keypointnet_ros/keypointnet_ros/scripts')
#import os
#import logging 
#from datetime import datetime

# cv and visualization modules
# import cv2
#import random
import numpy as np
#import pandas as pd
#import matplotlib.pylab as plt  # For interactive coding, e.g., Notebook
# import matplotlib.pyplot as plt # For (non-interactive) scripts

### learning modules
import paddle
#import paddle.nn as nn
#from paddle.vision.models import resnet50, resnet101, resnet152
#from paddle.io import Dataset
#from sklearn.model_selection import train_test_split
#from sklearn.metrics.pairwise import euclidean_distances 
#from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix

### custom modules
import inference.transforms as trans 
#import transforms as trans 
#import functional as F
# The transformations for KeypointD and Classification is different, so use seperate Datasets
from utils.PCDataset import img_dataset # Dataset of Direct state Classification
# import utils.PCDataset.img_dataset as img_dataset
from utils.KPDataset import imgDataset  # Dataset of KeyPoint Detection
#from utils.KPmetrics import cal_ed, cal_ed_val, cal_loss # Metrics of Keypoint Detection
#from models.resnet34_classification_paddle import Model_resnet34
#from models.keypointnet_deepest_paddle import KeypointNet_Deepest # GResNet-Deepest
from inference.config import image_size, pad_total, label_file, infer_mode, shuffle_key,num_workers_key #test_filelists2, best_PCmodel_path,best_PCmodel_path2,best_KPmodel_path 
from inference.orientation_cal import img_to_object_r_coor, yaw_cal
from sklearn.metrics.pairwise import euclidean_distances

### get dateloader
def get_dataloader(test_filelists, dataset_, transforms_, label_file_, infer_mode_, batchsize_, shuffle_, num_workers_):
    test_dataset = dataset_(image_file = test_filelists,
                        # dataset_root=testset_root, 
                        img_transforms=transforms_,
                        label_file=label_file_,
                        mode=infer_mode_)   #  ShoesStatesTrainingGT.xls

    test_loader = paddle.io.DataLoader(
        test_dataset,
        batch_sampler=paddle.io.DistributedBatchSampler(test_dataset, batch_size=batchsize_, shuffle=shuffle_, drop_last=False),
        num_workers=num_workers_,
        return_list=True,
        use_shared_memory=False
    )
    return test_loader

### get trained model
def get_trained_model(model, model_path): 
    para_state_dict = paddle.load(model_path)
    model.set_state_dict(para_state_dict)
    return model

## Direct State Classification
### Direct State Classification inference function 
def PCinfer(model1,model2, test_filelists2):  # model3,
    '''
    ### Direct State Classification dataloader v1
    PC_transforms = trans.Compose([
        trans.PaddedSquare('constant'),  # may use 'edge' mode when testing in real scenes.  'constant'    
        trans.Resize((image_size//2, image_size//2))
    ])
    PC_test_dataset = img_dataset(image_file = test_filelists2,
                            # dataset_root=testset_root, 
                            img_transforms=PC_transforms,
    #                        label_file="ShoesStatesTestingGT.xls",
                            mode=infer_mode)   #  ShoesStatesTrainingGT.xls
    
    PC_test_loader = paddle.io.DataLoader(
        PC_test_dataset,
        batch_sampler=paddle.io.DistributedBatchSampler(PC_test_dataset, batch_size=batchsize, shuffle=False, drop_last=False),
        # num_workers=num_workers,
        return_list=True,
        use_shared_memory=False
    )
    '''
    batchsize = len(test_filelists2)
    ### Direct State Classification dataloader v2
    PC_transforms = trans.Compose([
        trans.PaddedSquare('constant'),  # may use 'edge' mode when testing in real scenes.  'constant'    
        trans.Resize((image_size//2, image_size//2))
        ])
    PC_test_loader = get_dataloader(test_filelists2, img_dataset,PC_transforms, label_file, infer_mode, batchsize, shuffle_key, num_workers_key)
    
    model1.eval()
    if model2 != None:
        model2.eval()

    with paddle.no_grad():
        for data in PC_test_loader:
            imgs = (data[0] / 255.).astype("float32")
            # data[1]
            
            if model2 == None:
                logits = model1(imgs)
            else: ### Ensemble 2 ResNet34 nets
                logits1 = model1(imgs)   # 
                logits2 = model2(imgs)
                logits = 0.5*logits1 +0.5*logits2   # best: 0.5*0.9812 + 0.5*0.9801
            # print(logits)  # shape=[batchsize , 3]

    return logits.numpy().argmax(1) # 0,1,2 stands for top, side, bottom respectively


## Keypoints Detection
### KP inference function
def KPinfer(model, test_filelists2, state_mode = False, orient_mode = False): # state_mode and orient_mode are post processing mode flag
    '''
    ### Keypoint dataloader v1
    KP_transforms = trans.Compose([
        trans.PaddedSquare('constant'),  # may use 'edge' mode when testing in real scenes. 
        trans.RandomPadWithoutPoints(pad_thresh_l=pad_total, pad_thresh_h=pad_total)  # accroding to CropbyBBxinDarknet; # accroding to CropbyBBxinDarknet  pad_thresh_l=0.316, pad_thresh_h=0.412
    ])
    
    KP_test_dataset = imgDataset(image_file = test_filelists2,  # test_filelists2[0:2] images collected from real scenes
                            # pred_xy=pred_xy,
                           img_transforms = KP_transforms,
                        #    label_file=None,
                           mode=infer_mode)
    
    KP_test_loader = paddle.io.DataLoader(
        KP_test_dataset,
        batch_sampler=paddle.io.DistributedBatchSampler(KP_test_dataset, batch_size=batchsize, shuffle=False, drop_last=False),
        # num_workers=num_workers,
        return_list=True,
        use_shared_memory=False
    )
    '''
    batchsize = len(test_filelists2)
    ### Keypoint dataloader v2
    KP_transforms = trans.Compose([
        trans.PaddedSquare('constant'),  # may use 'edge' mode when testing in real scenes. 
        trans.RandomPadWithoutPoints(pad_thresh_l=pad_total, pad_thresh_h=pad_total)  # accroding to CropbyBBxinDarknet; # accroding to CropbyBBxinDarknet  pad_thresh_l=0.316, pad_thresh_h=0.412
    ])
    KP_test_loader = get_dataloader(test_filelists2, imgDataset, KP_transforms, label_file, infer_mode, batchsize, shuffle_key, num_workers_key)
    
    model.eval()
    logits = []
    imgs = []
    # cache = []  # ?
    with paddle.no_grad(): 
        # for img, idx, h, w, fh, fw in KP_test_dataset:   # h,w is the original size of img images
        #     img = img[np.newaxis, ...]    
        #     img = paddle.to_tensor((img / 255.).astype("float32"))    
        #     logit = model(img)
            
        #     logits.append(logit)
        #     imgs.append(img)
        #     # pred_coor = logits.numpy()
        for data in KP_test_loader:  #  img, idx, h, w, fh, fw
            imgs = (data[0] / 255.).astype("float32")
#            fhs = data[4].astype('int8') # fhs = data[4].astype('int8')[0]
#            fws = data[5].astype('int8')
            fhs = data[4]
            fws = data[5]
            logits = model(imgs)
            
            # post-processing - from heatmaps to keypoints       
            confident_kps = np.zeros((logits.shape[0],logits.shape[1]*3))  # (logits.shape[0],15) keypoints with confidence, i.e., [toe_c, toe_x, toe_y, heel_c ...,  inside ..., outside ..., topline ... ] 
            h= logits.shape[2]
            w= logits.shape[3]
            
            ## yaws list for Post processing - calculate orientations
            orientations = np.zeros((logits.shape[0],1)) # yaws
            ## states list
            states = np.zeros(logits.shape[0]).astype(int)
            kp_thresh = 0.794
            ed_c = 0.855

            for i in range(logits.shape[0]):  # batchsize
                for j in range(logits.shape[1]):  # logit.shape[1]==5
                    #if logits[i][j].max() >= 0.5: # 0.794 for TestingSet
                    kc_p= logits[i][j].max()
                    kx_p= np.where(logits[i][j]==logits[i][j].max())[1][0]  # predicted keypoints
                    ky_p= np.where(logits[i][j]==logits[i][j].max())[0][0]
                    kx_p, ky_p= kx_p/w, ky_p/h # normalized kx, ky
        
                    ### Transfer from current coordinate system to OD cropped img coordinate system 
                    # Inference should implement the reverse process of data augmentation (KP_transforms):
                    # Resize didn't change the relative position
                    # Reverse RandomPadWithoutPoints(pad_total) (side1 → side0) by minus pad_left1 or pad_top1
                    # Reverse Processing of PaddedSquare (side0→fw,fh) by minus pad_left0 or pad_top0
                    fh, fw =  fhs[i], fws[i] # fws[i][0]
                    s0 = max(fh,fw)  # length of side0, fh, fw is the shape of original images cropped from Object Detection
                    if fh<=fw:
                        pad_top0 = int((s0 - fh)/2)
                        pad_left0=0
                    else: # fh > fw
                        pad_top0 = 0
                        pad_left0 = int((s0 - fw)/2)
                    
                    delta_s1 = int(s0*pad_total)
                    s1=s0+delta_s1   # length od side1 
                    pad_top1 = delta_s1//2
                    pad_left1 = delta_s1//2
                    fx = kx_p*s1 - pad_left1 - pad_left0
                    fy = ky_p*s1 - pad_top1  - pad_top0
                    
                    confident_kps[i][3*j] = kc_p
                    confident_kps[i][3*j+1] = fx 
                    confident_kps[i][3*j+2] = fy
                
                if orient_mode == True:
                    ## Post processing - calculate orientations, including yaw 
                    kx_toe=  np.where(logits[i][0]==logits[i][0].max())[1][0]  # predicted toe keypoint
                    ky_toe=  np.where(logits[i][0]==logits[i][0].max())[0][0]
                    kx_heel= np.where(logits[i][1]==logits[i][1].max())[1][0]  # predicted heel keypoint
                    ky_heel= np.where(logits[i][1]==logits[i][1].max())[0][0]
        
                    Toe  = img_to_object_r_coor(w,h,kx_toe, ky_toe)
                    Heel = img_to_object_r_coor(w,h,kx_heel,ky_heel)
                    HT = Toe - Heel
                    # HT0 =  np.array([0, 10]) # (z1, x1)
                    HT0 =  np.array([10, 0]) # (x1, y1)
                    yaw_pred = yaw_cal(HT0, HT)
                    orientations[i][0]= yaw_pred
                    ##
                
                if state_mode == True:
                    kp_peaks = np.array([logits[i][0].max(), logits[i][1].max(), logits[i][2].max(),logits[i][3].max(),logits[i][4].max()]) # the heatmap max values of 5 keypoints at toe, heel, inside, outside, topline
                    # method 6: kp_thresh + argmin + relative euclidean_distance # optimal ed_c=0.855, kp_thresh = 0.794 for method 6, then macro_precision=0.9547 weighted_precision=0.9457
                    kx_i= np.where(logits[i][2]==logits[i][2].max())[1][0]  # predicted inside keypoint
                    ky_i= np.where(logits[i][2]==logits[i][2].max())[0][0]
                    kx_o= np.where(logits[i][3]==logits[i][3].max())[1][0]  # predicted outside keypoint
                    ky_o= np.where(logits[i][3]==logits[i][3].max())[0][0]
                
                    # calculate euclidean_distances by normalizad kx, ky
                    ed_io = euclidean_distances([[kx_i/w, ky_i/h]],[[kx_o/w, ky_o/h]]) # 2D arrays are expected 
                    
                    kx_toe= np.where(logits[i][0]==logits[i][0].max())[1][0]  # predicted toe keypoint
                    ky_toe= np.where(logits[i][0]==logits[i][0].max())[0][0]
                    ed_it = euclidean_distances([[kx_i/w, ky_i/h]],[[kx_toe/w, ky_toe/h]]) # 2D arrays are expected 
                    ed_ot = euclidean_distances([[kx_o/w, ky_o/h]],[[kx_toe/w, ky_toe/h]]) # 2D arrays are expected
                    ed_thresh = min(ed_it, ed_ot)*ed_c # ed_c coefficient
                    
                    if kp_peaks.min() < kp_thresh:
                        if kp_peaks.argmin() == 4:
                            states[i]= 2 # 2 stands for bottom 
                        elif kp_peaks.argmin() == 3 or kp_peaks.argmin() == 2:
                            states[i]= 1 # 1 stands for side
                        else:  # kp_peaks.argmin() == 0 or kp_peaks.argmin() == 1 
                            states[i]= 0 
                    else:  # kp_peaks.min() > kp_thresh or  kp_peaks.min() = kp_thresh
                        if ed_io < ed_thresh:  
                            states[i]= 1 # 1 stands for side
                        else: # ed_io >= ed_thresh
                            states[i]= 0  # 0 stands for top
                    
                
    if orient_mode == True and state_mode == True:
        return confident_kps, orientations, states
    elif orient_mode == True and state_mode == False:
        return confident_kps, orientations
    elif orient_mode == False and state_mode == True:
        return confident_kps, states
    else: # orient_mode == False and state_mode == False:
        return confident_kps
        
    # return confident_kps, orientations # KPlogits, KPimgs, confident_kps


'''
## Call PCinfer(), input test_filelists2, output state_classes
def CallPCinfer(PCmodel, PCmodel2, test_filelists2):

    return state_classes

## Call KPinfer(), input test_filelists2, output confident_kps
def CallKPinfer(test_filelists2):

    return confident_kps
'''


#import argparse
#def parse_args():
#    parser = argparse.ArgumentParser(description='Train network')
#
#    # Network
#    parser.add_argument('--network', type=str, default='grconvnet3',
#                        help='Network name in inference/models')
#    args = parser.parse_args()
#    return args

