#!/home/dongyi/anaconda3/envs/paddle_env/bin/python3
#import os
import numpy as np
import cv2
#import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import cohen_kappa_score
#from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix # plot_precision_recall_curve
import paddle
#import paddle.nn as nn
#import paddle.nn.functional as F
#import transforms as trans
import warnings
warnings.filterwarnings('ignore')

class img_dataset(paddle.io.Dataset):
    """
    getitem() output
    	img: RGB uint8 image with shape (3, image_size, image_size)
    """
    def __init__(self,
                 img_transforms,
                 image_file,  # the absolute path of images
                 # dataset_root,
                 label_file='',
                 filelists=None,
                 num_classes=3,
                 mode='train'):

        self.image_file = image_file
        image_idxs = self.image_file  # image_idxs and self.image_file is images list when self.mode == "deploy"
        self.img_transforms = img_transforms
        self.mode = mode.lower()
        self.num_classes = num_classes
        
        if self.mode == 'train':
            label = {row['img_name']: row[1:].values 
                        for _, row in pd.read_excel(label_file).iterrows()}  # dictionary {image name: label}
            # self.file_list = [[f, label[f]] for f in os.listdir(dataset_root)]
            self.file_list = [[image_idxs[i], label[image_idxs[i].split('/')[-1]]] for i in range(len(image_idxs))]
        elif self.mode == "test":
            # self.file_list = [[f, None] for f in os.listdir(dataset_root)]
            self.file_list = [[image_idxs[i], None] for i in range(len(image_idxs))]
        
        elif self.mode == "deploy":
            self.file_list = [[image_idxs[i], None] for i in range(len(image_idxs))] # image_idxs[i] is an image when self.mode == "deploy"
        
        if filelists is not None:
            self.file_list = [item for item in self.file_list if item[0] in filelists]

    def __getitem__(self, idx):
        if self.mode == "deploy":
            img0, label = self.file_list[idx] # image_idxs[i] is an image when self.mode == "deploy"
            img = img0[:, :, ::-1] # BGR -> RGB
        else: # self.mode == "train" or "test"
            real_index, label = self.file_list[idx] 
            img_path  =  real_index  # real_index is the abosolute path
            # img_path = os.path.join(self.dataset_root, real_index) 
            img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
    
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        # normlize on GPU to save CPU Memory and IO consuming.
        # img = (img / 255.).astype("float32")

        img = img.transpose(2, 0, 1) # H, W, C -> C, H, W

        if self.mode == 'deploy':
            return img, idx
        elif self.mode == 'test':
            return img, real_index
        elif self.mode == "train":
            label = label.argmax()
            return img, label
        else:
            print('No such mode')
            return None


    def __len__(self):
        return len(self.file_list)
