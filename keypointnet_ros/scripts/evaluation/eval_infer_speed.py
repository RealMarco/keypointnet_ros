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
import sys
sys.path.append('/home/marco/catkin_workspace/src/keypointnet_ros/keypointnet_ros/scripts')
# sys.path.append('/home/dongyi/ur_ws_vision/src/keypointnet_ros/keypointnet_ros/scripts')
# sys.path.append('/home/dongyi/anaconda3/envs/paddle_env/lib/python3.9/site-packages')
if '/usr/lib/python3/dist-packages' in sys.path: # before importing other modules or packages
    sys.path.remove('/usr/lib/python3/dist-packages')
print (sys.path)

# cv and visualization modules
import cv2
import time
import os
#import random
#import numpy as np
#import pandas as pd
#import matplotlib.pylab as plt  # For interactive coding, e.g., Notebook
# import matplotlib.pyplot as plt # For (non-interactive) scripts

### custom modules
from models.resnet34_classification_paddle import Model_resnet34
from models.keypointnet_deepest_paddle import KeypointNet_Deepest # GResNet-Deepest
from inference.config import test_filelists2, infer_mode, best_PCmodel_path,best_PCmodel_path2,best_KPmodel_path #,PC_transforms,KP_transforms
from inference.keypoints_eval import get_trained_model, PCinfer, KPinfer

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
    testset_root = "../dataset/ShoesKeypoints_ValidationSet/"
    if infer_mode == 'test':     # test_filelists2 is an image name list when self.mode == "test"
        test_filelists2 = [os.path.join(testset_root, i) for i in os.listdir(testset_root)] 
        # test_filelists2 = ['dataset/ShoesKeypoints_ValidationSet/IMG_20210302_151345.jpg', 'dataset/ShoesKeypoints_ValidationSet/IMG_20211207_114000.jpg']  # 
    elif infer_mode == 'deploy': # test_filelists2 is an image list when self.mode == "deploy"
        filelists2 = [os.path.join(testset_root, i) for i in os.listdir(testset_root)]
        test_filelists2 =  [cv2.imread(i) for i in filelists2]
        # test_filelists2 =  [cv2.imread(i) for i in ['dataset/ShoesKeypoints_ValidationSet/IMG_20211207_144827_1.jpg', 'dataset/ShoesKeypoints_ValidationSet/IMG_20210302_155456.jpg']]
    else:
        print('The mode %s is to be updated soon.'%infer_mode)
    
    if len(test_filelists2) < 8:
        batchsize = len(test_filelists2)
    else:
        batchsize = 8

    ### Loading Direct State Classification model v2
    PCmodel = Model_resnet34()
    PCmodel = get_trained_model(PCmodel, best_PCmodel_path)
    if best_PCmodel_path2 == '' or best_PCmodel_path2 == None:
        PCmodel2= None
    else:
        PCmodel2= Model_resnet34()
        PCmodel2= get_trained_model(PCmodel2,best_PCmodel_path2)
        
    ### Loading keypoint detection model v2
    KPmodel= KeypointNet_Deepest(input_channels=3, output_channels=5, channel_size=32,dropout=True, prob=0)
    KPmodel= get_trained_model(KPmodel, best_KPmodel_path)
    
    
    ### Infer state class
    # start1 =  time.clock()  ### time.clock() calculates the running time of the CPU
    start1 = time.time() ### time.time() calculates the running time of the whole program, including the time of CPU, i/o, sleep.
    state_classes =PCinfer(PCmodel,PCmodel2, test_filelists2, batchsize)
    # end1 = time.clock()
    end1 = time.time()
    print("Average inference time for DL-based pose classfication: %f"%((end1 -start1)/len(test_filelists2)))
    
    ### Infer keypoints with confidence in the format of [toe_c, toe_x, toe_y, heel_c ...,  inside ..., outside ..., topline ... ] 
    # start2 = time.clock() ### time.clock() calculates the running time of the CPU
    start2 = time.time() ### time.time() calculates the running time of the whole program, including the time of CPU, i/o, sleep.
    confident_kps, orientations, states= KPinfer(KPmodel,test_filelists2, batchsize, state_mode = True, orient_mode =True)  #the size of confident_kps is CroppeedImg num x 15
    #confident_kps, states= KPinfer(KPmodel,test_filelists2, batchsize, state_mode = True, orient_mode =False)
    # confident_kps, orientations= KPinfer(KPmodel,test_filelists2, batchsize, state_mode = False, orient_mode =True)
    # confident_kps = KPinfer(KPmodel,test_filelists2, batchsize, state_mode = False, orient_mode =False) 
    # end2 = time.clock()
    end2 = time.time()
    print("Average inference time for keypoint detection+post-proscessing: %f"%((end2 -start2)/len(test_filelists2)))
    # print(len(test_filelists2))

    """
    ### Visualizing predicted Keypoints
    # plt.figure(figsize=(20, 20)) # , for matplotlib
    colors =[(136,32,29), (0,0,192), (160,48,112),(171,171,175),(233,44,242)] # BGR
    # colors = ['#1D2088', '#C00000','#7030A0','#AFABAB', '#F22CE9'] # blue, red, purple, grey, magenta, , for matplotlib
    # For RGB, [(29,32,136), (192,0,0), (112,48,160),(175,171,171),(242,44,233)]
    # For BGR, [(136,32,29), (0,0,192), (160,48,112),(171,171,175),(233,44,242)]
    shoe_states= ['top','side','bottom'] # 0,1,2 represents top, side, bottom state respectively
    keypoint_classes =  ['toe','heel','inside','outside','topline']
    for i in range(batchsize):
        if infer_mode == 'test':
            img =  cv2.imread(test_filelists2[i])[:, :, ::-1]
        elif infer_mode == 'deploy':
            img= test_filelists2[i] # [:, :, ::-1] # BGR -> RGB, , for matplotlib
        else:
            print('The model %s is to be updated soon'%infer_mode)
        
        '''
        # Matplotlib visulization
        plt.subplot(1,batchsize,i+1)
        plt.imshow(img)
        for j in range(confident_kps.shape[1]//3):
            if confident_kps[i][3*j]>0.5:
                plt.plot(confident_kps[i][3*j+1]+0.5,confident_kps[i][3*j+2]+0.5,'.', color=colors[j], markersize=16) # +0.5 means centre of pixel
                plt.text(confident_kps[i][3*j+1]+0.5,confident_kps[i][3*j+2]+0.5, "({:.2f},{:.2f})".format(confident_kps[i][3*j+1],confident_kps[i][3*j+2]))
                
        plt.text(10, 30, 'Direct Classification: %s'%shoe_states[state_classes[i]])
        plt.text(10, 50, 'KP-based Classification: %s'%shoe_states[states[i]])
        plt.text(10, 70, 'alpha=%.2f'%orientations[i][0]) # str(round(orientations[i][0],2))
        '''
        # opencv visualization
        for j in range(confident_kps.shape[1]//3):
            if confident_kps[i][3*j]>0.5:
                img =cv2.circle(img, (round(confident_kps[i][3*j+1]), round(confident_kps[i][3*j+2])), 8, colors[j], -1)
                img=cv2.putText(img, keypoint_classes[j], (round(confident_kps[i][3*j+1]), round(confident_kps[i][3*j+2])), cv2.FONT_HERSHEY_SIMPLEX , 1, colors[j], 2, cv2.LINE_AA)
        img=cv2.putText(img, shoe_states[state_classes[i]], (10, 30), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 2, cv2.LINE_AA)
        img=cv2.putText(img, shoe_states[states[i]], (10, 70), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 2, cv2.LINE_AA)
        img=cv2.putText(img, str(round(orientations[i][0],2)), (10, 110), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 2, cv2.LINE_AA)
        # print(type(img))
        #cv2.namedWindow("image %d"%i)
        # cv2.imshow("image %d"%i, img)
        cv2.imshow("image",img)
        #cv2.waitKey(4000)
        while (cv2.waitKey(4000)==27):
            pass
        #cv2.destroyAllWindows()
        ### TO DO -  state class-based keypoint outputs
    
    """
