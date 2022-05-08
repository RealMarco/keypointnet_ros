#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: marco
"""

### import necessary packages
### system modules
#import sys 
#sys.path.append('/home/aistudio/external-libraries')
#import os
#import logging 
#from datetime import datetime

# cv and visualization modules
import cv2
#import random
import numpy as np
#import pandas as pd
import matplotlib.pylab as plt  # For interactive coding, e.g., Notebook
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
import transforms as trans 
#import functional as F
# The transformations for KeypointD and Classification is different, so use seperate Datasets
from utils.PCDataset import img_dataset # Dataset of Direct state Classification
from utils.KPDataset import imgDataset  # Dataset of KeyPoint Detection
#from utils.KPmetrics import cal_ed, cal_ed_val, cal_loss # Metrics of Keypoint Detection
from models.resnet34_classification_paddle import Model_resnet34
from models.keypointnet_deepest_paddle import KeypointNet_Deepest # GResNet-Deepest
from inference.config import image_size, test_filelists2, pad_total, infer_mode, shuffle_key,num_workers_key, best_PCmodel_path,best_PCmodel_path2,best_KPmodel_path #,PC_transforms,KP_transforms

### get dateloader
def get_dataloader(test_filelists, dataset_, transforms_, infer_mode_, batchsize_, shuffle_, num_workers_):
    test_dataset = dataset_(image_file = test_filelists,
                        # dataset_root=testset_root, 
                        img_transforms=transforms_,
#                        label_file="ShoesStatesTestingGT.xls",
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
    ### Direct State Classification dataloader v2
    PC_transforms = trans.Compose([
        trans.PaddedSquare('constant'),  # may use 'edge' mode when testing in real scenes.  'constant'    
        trans.Resize((image_size//2, image_size//2))
        ])
    PC_test_loader = get_dataloader(test_filelists2, img_dataset,PC_transforms, infer_mode, batchsize, shuffle_key, num_workers_key)
    
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
def KPinfer(model, test_filelists2): #KP_test_dataset
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
    ### Keypoint dataloader v2
    KP_transforms = trans.Compose([
        trans.PaddedSquare('constant'),  # may use 'edge' mode when testing in real scenes. 
        trans.RandomPadWithoutPoints(pad_thresh_l=pad_total, pad_thresh_h=pad_total)  # accroding to CropbyBBxinDarknet; # accroding to CropbyBBxinDarknet  pad_thresh_l=0.316, pad_thresh_h=0.412
    ])
    KP_test_loader = get_dataloader(test_filelists2, imgDataset, KP_transforms, infer_mode, batchsize, shuffle_key, num_workers_key)
    
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
            confident_kps = np.zeros((logits.shape[0],15))  # keypoints with confidence, i.e., [toe_c, toe_x, toe_y, heel_c ...,  inside ..., outside ..., topline ... ] 
            h= logits.shape[2]
            w= logits.shape[3]
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
    
    return confident_kps # KPlogits, KPimgs, confident_kps

'''
## Call PCinfer(), input test_filelists2, output state_classes
def CallPCinfer(PCmodel, PCmodel2, test_filelists2):

    return state_classes

## Call KPinfer(), input test_filelists2, output confident_kps
def CallKPinfer(test_filelists2):

    return confident_kps
'''

## make Keypoints.msg, Keypoint.msg (confidence/probability,x,y,String state), ShoeStates.msg/ShoePoseClass.msg in KD_ros/KD_ros_msgs/msg
### considering the extensibility, use a keypoints[] class array instead of an array/list consists of all the info of keypoints 

#import argparse
#def parse_args():
#    parser = argparse.ArgumentParser(description='Train network')
#
#    # Network
#    parser.add_argument('--network', type=str, default='grconvnet3',
#                        help='Network name in inference/models')
#    args = parser.parse_args()
#    return args


if __name__ == '__main__': # avoid automatic running below lines when this .py file is imported by others.    
    ### image subscriber 
    if infer_mode == 'test':     # test_filelists2 is an image name list when self.mode == "test"
        test_filelists2 = ['TestingSet/IMG_20210302_151345.jpg', 'TestingSet/IMG_20211207_114000.jpg']
    elif infer_mode == 'deploy': # test_filelists2 is an image list when self.mode == "deploy"
        test_filelists2 =  [cv2.imread(i) for i in ['TestingSet/IMG_20210302_151345.jpg', 'TestingSet/IMG_20211207_114000.jpg']]
    else:
        print('The model %s is to be updated soon.'%infer_mode)
        
    batchsize = len(test_filelists2)
    
    
    '''
    ### Loading Direct State Classification model v1
    best_PCmodel_path  = "trained_models/ShoePoseClassificationResNet/PC_ResNet34_0.9812.pdparams" # 0.5*0.9812 + 0.5*0.9801
    best_PCmodel_path2 = "trained_models/ShoePoseClassificationResNet/PC_ResNet34_0.9801.pdparams" # 0.5*0.9812 + 0.5*0.9801 
    PCmodel = Model_resnet34()
    PC_para_state_dict = paddle.load(best_PCmodel_path)
    PCmodel.set_state_dict(PC_para_state_dict)
    # PCmodel2= None
    PCmodel2= Model_resnet34()
    PC_para_state_dict2 = paddle.load(best_PCmodel_path2)
    PCmodel2.set_state_dict(PC_para_state_dict2)
    '''
    ### Loading Direct State Classification model v2
    PCmodel = Model_resnet34()
    PCmodel = get_trained_model(PCmodel, best_PCmodel_path)
    if best_PCmodel_path2 == '' or best_PCmodel_path2 == None:
        PCmodel2= None
    else:
        PCmodel2= Model_resnet34()
        PCmodel2= get_trained_model(PCmodel2,best_PCmodel_path2)
    
    ### Infer state class
    state_classes =PCinfer(PCmodel,PCmodel2, test_filelists2)
    
    
    '''
    ### Loading keypoint detection model v1
    KPmodel= KeypointNet_Deepest(input_channels=3, output_channels=5, channel_size=32,dropout=True, prob=0)
    best_KPmodel_path = "trained_models/8_3Heatmap_GResNet-Deepest_C32_no_TestSet_finetune_220315_1219/best_model_0.016625/model.pdparams"
    #best_KPmodel_path = "trained_models/2DShoeKeypointDetection/2DKeypointNet_0.016625.pdparams"
    KP_para_state_dict = paddle.load(best_KPmodel_path)
    KPmodel.set_state_dict(KP_para_state_dict)
    '''
    ### Loading keypoint detection model v2
    KPmodel= KeypointNet_Deepest(input_channels=3, output_channels=5, channel_size=32,dropout=True, prob=0)
    KPmodel= get_trained_model(KPmodel, best_KPmodel_path)
    
    ### Infer keypoints with confidence in the format of [toe_c, toe_x, toe_y, heel_c ...,  inside ..., outside ..., topline ... ] 
    confident_kps= KPinfer(KPmodel,test_filelists2)  #the size of confident_kps is CroppeedImg num x 15
    

    ### Visualizing predicted Keypoints
    plt.figure(figsize=(20, 20))
    colors = ['#1D2088', '#C00000','#7030A0','#AFABAB', '#F22CE9'] # blue, red, purple, grey, magenta
    # For RGB, [(29,32,136), (192,0,0), (112,48,160),(175,171,171),(242,44,233)]
    # For BGR, [(136,32,29), (0,0,192), (160,48,112),(171,171,175),(233,44,242)]
    for i in range(batchsize):
        if infer_mode == 'test':
            img =  cv2.imread(test_filelists2[i])[:, :, ::-1]
        elif infer_mode == 'deploy':
            img= test_filelists2[i][:, :, ::-1] # BGR -> RGB
        else:
            print('The model %s is to be updated soon'%infer_mode)
            
        plt.subplot(1,batchsize,i+1)
        plt.imshow(img)
        for j in range(confident_kps.shape[1]//3):
            if confident_kps[i][3*j]>0.5:
                plt.plot(confident_kps[i][3*j+1]+0.5,confident_kps[i][3*j+2]+0.5,'.', color=colors[j], markersize=16) # +0.5 means centre of pixel
                plt.text(confident_kps[i][3*j+1]+0.5,confident_kps[i][3*j+2]+0.5, "({:.2f},{:.2f})".format(confident_kps[i][3*j+1],confident_kps[i][3*j+2]))
        ### TO DO -  state class-based keypoint outputs
    ### Outputing predicted state
    states= ['top','side','bottom'] # 0,1,2 represents top, side, bottom state respectively
    for i in range(batchsize):
        print(states[state_classes[i]])