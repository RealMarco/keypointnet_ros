# KeypointNet: Keypoints Detection, Keypoing-based Pose Classfication and Orientation Estimation
- keypointnet_ros consists of core code of Keypoints Detection, Keypoing-based Pose Classfication (a.k.a. States CLassification) and Orientation Estimation for training and ROS deployment. 
- keypointnet_ros_msgs is messages used for keypoint_ros

## Overview  
## Citing  
## Installation  
### Dependencies  
### Building (Compiling)  



## Files Description  
1. [/keypointnet_ros/scripts/](keypointnet_ros/scripts/) includes python scripts   
    1. [KeypointDetector.py](keypointnet_ros/scripts/KeypointDetector.py) works in ROS environment and achieves fucntions below by multi-threads  
        - Subscribing msgs in topic '/camera/color/image_raw', '/darknet_ros/bounding_boxes', '/darknet_ros/detection_image'  
        - Inferring the keypoints, states, (poses) of raw camera images by call functions in keypoints_pred.py  
        - Publishing msgs in topic '/keypointnet_ros/state_keypoints', '/keypointnet_ros/keypoint_image' 
    2. [keypoints_test.py](keypointnet_ros/scripts/keypoints_test.py) works in python environment and tests the keypoint detection performance by a small batch of images.
    3. [keypoints_train.py](keypointnet_ros/scripts/keypoints_train.py) works in python environment and trains the keypoint detection model according to config.py.
    4. [keypoints_pred.py](keypointnet_ros/scripts/inference/keypoints_pred.py) can accomplish the inference of keypoints and states in deployment and testing segment. It works for KeypointDetector.py anf keypoints_test.py. 
    5. [/inference/config.py](keypointnet_ros/scripts/inference/config.py) is the configuration file of training and deployment our keypoint detection, which should be input and adjusted by users.  
    6. [/utils/KPDataset.py](keypointnet_ros/scripts/utils/KPDataset.py) and [/utils/PCDataset.py](keypointnet_ros/scripts/utils/PCDataset.py) read and augment date, then generate dataset classes to make them understandable for PaddlePaddle.  
    7. [/models/](keypointnet_ros/scripts/models/) and /trained_models/ provide network (model) architecture and pre-trained weights respectively.  
    8. [transforms.py](keypointnet_ros/scripts/transforms.py) with [functional.py](keypointnet_ros/scripts/functional.py) contains many useful image data augmentation (transformation) methods, which were not only designed for this task, but also suitable for other image-based Machine Learning tasks (i.e., classification, segmentation, regression and keypoint detection). They are not framework-limited, and it means that any python-based framework like PyTorch, PaddlePaddle can use them.  
    9. [2DShoesKeypointDetection_clear_version.ipynb](keypointnet_ros/scripts/2DShoesKeypointDetection_clear_version.ipynb) is the jupyter notebook version of our code.
2. [/keypointnet_ros/src/](keypointnet_ros/scripts/keypointnet_ros/src/) includes cpp code

## To Be Uptated
1. Keypoint-based states classification
2. Keypoint-based Pose Estimation

## To Do
11. Train resnet34 classification model without @paddle.jit.to_static to acuquire a model with smaller size    
12. Table. 2. To compare shoe keypoint detection algorithms (precision, sample efficiency (the size of the dataset), inference speed)?   

## Start to run the keypointnet_ros
    $ catkin_make 
    #$ catkin_make -DPYTHON_EXECUTABLE=/home/dongyi/anaconda3/envs/paddle_env/bin/python 
    #$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 
    
    $ source devel/setup.bash  
    
    #$ roslaunch realsense2_camera rs_camera.launch 
    $ roslaunch realsense2_camera rs_camera.launch align_depth:=true 
    
    $ roslaunch darknet_ros yolov4-p5RP.launch   
    
    # $ roscore # $ roslauch calls ros master automatically, thus no need to roscore again.   
    $ rosrun keypointnet_ros KeypointDetector.py  

## Call keypoint_ros in robot system (To Be Validated and Improved)     
    $ roslaunch realsense2_camera rs_camera.launch align_depth:=true  
    
    # run the calibration software  
    $ roslaunch easy_handeye eye_to_hand_calibration.launch  
    
    $ roslaunch darknet_ros yolov4-p5RP.launch
    $ rosrun keypointnet_ros KeypointDetector.py
    $ rosrun ur_smach visionInterface.py
    ...  
