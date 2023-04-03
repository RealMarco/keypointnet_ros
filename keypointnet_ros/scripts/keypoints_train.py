#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#!/home/dongyi/anaconda3/envs/paddle_env/bin/python
Created on Tue Aug 30 12:51:52 2022

@author: marco
"""
import sys
sys.path.append('/home/marco/robotic_sorting/src/keypointnet_ros/keypointnet_ros/scripts')
# sys.path.append('/home/dongyi/anaconda3/envs/paddle_env/lib/python3.9/site-packages')
if '/usr/lib/python3/dist-packages' in sys.path: # before importing other modules or packages
    sys.path.remove('/usr/lib/python3/dist-packages')
print (sys.path)
import os
import numpy as np
# import logging
import datetime
import warnings

# learning modules
import paddle
from visualdl import LogWriter

### custom modules
import inference.transforms as trans
from models.keypointnet_deepest_paddle import KeypointNet_Deepest
from utils.KPDataset import imgDataset
from utils.KPmetrics import cal_loss, cal_ed_val
from inference.keypoints_pred import get_dataloader # , get_trained_model
from inference.config import iters, init_lr, optimizer_type, batchsize, pad_total, label_file, infer_mode, breakpoint_training, shuffle_key,num_workers_key, trainset_root, valset_root,best_KPmodel_path

# def train(model, iters, train_dataloader, val_dataloader, optimizer, scheduler,  log_interval, evl_interval, networkname):
def train(model, iters, train_dataloader, val_dataloader, optimizer, log_interval, evl_interval, networkname):
    iter = 0
    model.train()
    avg_loss_list = []
    avg_ED_list = []
    best_ED = sys.float_info.max

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')            # 
    net_desc = networkname+'_'+dt
    save_folder = os.path.join('trained_models', net_desc)
    logfolder = os.path.join(save_folder,'logs') 
    with LogWriter(logdir=logfolder) as writer: 
        while iter < iters:
            for img, lab in train_dataloader:
                iter += 1
                if iter > iters:
                    break
                imgs = (img / 255.).astype('float32')
                label = lab.astype("float32")

                logits = model(imgs)
                loss = cal_loss(logits, label)
                # print('loss in train',loss)

                for p,l in zip(logits.numpy(), label.numpy()):
                    avg_ED_list.append([p,l])
                
                # print('avg_ED_list', avg_ED_list)
                # optimizer.clear_grad() # when optimizer.XXX(parameters=model.parameters()), it is the same as model.clear_gradients()
                                         # it is quite slower than model.clear_gradients()
                loss.backward()
                optimizer.step()
                model.clear_gradients()  # In pyTorch, zero_grad()
                # scheduler.step(loss) # for optimizer.lr.ReduceOnPlateau
                # scheduler.step() # for optimizer.lr.PiecewiseDecay
                avg_loss_list.append(loss.numpy()[0])
                
                if iter % log_interval == 0:
                    avg_loss = np.array(avg_loss_list).mean()
                    # print(avg_loss)
                    avg_ED_list = np.array(avg_ED_list)
                    avg_ED = cal_ed_val(avg_ED_list[:, 0], avg_ED_list[:, 1]) # cal_ED
                    # print('ed in training', avg_ED)
                    avg_loss_list = []
                    avg_ED_list = []
                    
                    # logging.warning("[TRAIN] iter={}/{} avg_loss={:.5f} avg_ED={:.5f}".format(iter, iters, avg_loss, avg_ED[0][0]))
                    print("[TRAIN] iter={}/{} avg_loss={:.6f} avg_ED={:.6f}".format(iter, iters, avg_loss, avg_ED[0][0]))
                    writer.add_scalar(tag="train/avg_ED", step=iter, value=avg_ED[0][0])  # visualiztion
                    writer.add_scalar(tag="train/avg_loss", step=iter, value=avg_loss)  # visualiztion

                if iter % evl_interval == 0:
                    avg_loss, avg_ED = val(model, val_dataloader)
                    print("[EVAL] iter={}/{} avg_loss={:.6f} ED={:.6f}".format(iter, iters, avg_loss, avg_ED[0][0]))
                    writer.add_scalar(tag="eval/avg_ED", step=iter, value=avg_ED[0][0])  # visualiztion
                    writer.add_scalar(tag="eval/avg_loss", step=iter, value=avg_loss)  # visualiztion
                    if avg_ED <= best_ED : # metric is average ED
                    # if avg_ED <= 0.025 and avg_ED <= best_ED:         # change the saving condition to...
                        best_ED = avg_ED[0][0]
                        paddle.save(model.state_dict(),
                                os.path.join(save_folder,"best_model_{:.6f}".format(best_ED), 'model.pdparams'))
                        # paddle.save(optimizer.state_dict(), 
                                # os.path.join('trained_models',"best_model_{:.4f}".format(best_ED), 'optimizer.pdopt'))
                    model.train()

### validation function

def val(model, val_dataloader):
    model.eval()
    avg_loss_list = []
    cache = []
    with paddle.no_grad():
        for data in val_dataloader:
            imgs = (data[0] / 255.).astype("float32")
            labels = data[1].astype('float32')
            
            logits = model(imgs)
            for p, l in zip(logits.numpy(), labels.numpy()):
                cache.append([p, l])

            loss = cal_loss(logits, labels)
            avg_loss_list.append(loss.numpy()[0])

    cache = np.array(cache)
    ED = cal_ed_val(cache[:, 0], cache[:, 1])
    avg_loss = np.array(avg_loss_list).mean()

    return avg_loss, ED

if __name__ == '__main__': # avoid automatic running below lines when this .py file is imported by others.
    # Training
    train_filelists = [os.path.join(trainset_root, i) for i in os.listdir(trainset_root)]  # list the child directories of the trainset_root
    val_filelists = [os.path.join(valset_root,i) for i in  os.listdir(valset_root)]
    
    KP_transforms_train = trans.ComposeWithPoint([         
    trans.PaddedSquareWithPoints('constant'),  # may use 'edge' mode when testing in real scenes. 

    trans.RandomPadWithPoints(pad_thresh_l=0.416, pad_thresh_h=0.512),  # pad_rate should be larger than (√2-1) (0.414)
    trans.RandomHVFlipWithPoints(),
    trans.BackgroundReplacementWithPoints(),
    trans.RandomRotationWithPoints(180),
    trans.BackgroundReplacementWithPoints(),
    trans.ColorJitterWithPoints(0.2, 0.2, 0.2, 0.2)  # (brightness, contrast, saturation, hue)
])  
    KP_transforms_val = trans.ComposeWithPoint([
    # trans.ColorThresholdSegmentationWithPoints(color_mode='corner',hue_de=13,sat_de=107,val_de=110),
    trans.PaddedSquareWithPoints('constant'),  # may use 'edge' mode when testing in real scenes. 

    trans.RandomPadWithPoints(pad_thresh_l=pad_total, pad_thresh_h=pad_total),  #pad_thresh_l=0.326, pad_thresh_h=0.442 accroding to CropbyBBxinDarknet (1.04)  pad_thresh_l=0.400, pad_thresh_h=0.400
    # trans.ColorThresholdSegmentationWithPoints(color_mode='corner',hue_de=13,sat_de=107,val_de=110), # 13, 107, 140
    # trans.BackgroundReplacementWithPoints(default=True,r=36,g=146,b=145),
    # trans.ColorJitterWithPoints(0.1, 0.1, 0.1, 0.02)
])
    train_loader= get_dataloader(train_filelists, imgDataset, KP_transforms_train, label_file, infer_mode, batchsize, shuffle_key, num_workers_key)
    val_loader  = get_dataloader(val_filelists, imgDataset, KP_transforms_val, label_file, infer_mode, batchsize, shuffle_key, num_workers_key)
    
    # KeypointNet_Deepest
    model= KeypointNet_Deepest(input_channels=3, output_channels=5, channel_size=32,dropout=True, prob=0.1)
    model_info=paddle.summary(model,(1,3,512,512))  # (1,3,512,512) when image_size = 512
    print(model_info)
    
    # Training from breakpoint, Incremental Training, Transfer Learning
    if breakpoint_training == True: 
        para_state_dict = paddle.load(best_KPmodel_path)
        model.set_state_dict(para_state_dict)
    # Training from breakpoint
    
    if optimizer_type == "adam":
        optimizer = paddle.optimizer.Adam(init_lr, parameters=model.parameters())
    elif optimizer_type == 'sgd':
        optimizer = paddle.optimizer.SGD(learning_rate=init_lr, parameters=model.parameters(), 
                                        weight_decay=None) # , grad_clip=None, name=None
    else:
        print("No such optimzier, please add it by yourself")
    print(init_lr)
    
    ### training process    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train(model, iters, train_loader, val_loader, optimizer, log_interval=23, evl_interval=92, networkname = 'KeypointNet-Deepest_C32')
    
    ### save the parameters of optimizer
    # paddle.save(optimizer.state_dict(), os.path.join('trained_models',"SegHeatmap_nonkey_KeypointNet-Deepest_C24_no_TestSet_220714_1122", 'optimizer.pdopt'))
    

    
   
