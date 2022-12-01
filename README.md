# KeypointNet: Keypoints Detection, Keypoing-based Pose Classfication and Orientation Estimation
KeypointNet is a keypoint detection deep learning model for category-level semantic keypoint detection, with keypoint-based state classification and pose estimation (by post-processing). We  developed this easy-to-use keypointnet_ros package by integrating the training, testing, and deployment code, such that it can be applied in other ROS-based software system.  

The KeypointNet-ROS package has been tested under ROS Noetic and Ubuntu 20.04. This is research code, expect that it changes often and users make adapt some details for specific system and task.    

- keypointnet_ros consists of core code of Keypoints Detection, Keypoing-based Pose Classfication (a.k.a. States Classification) and Orientation Estimation for training and ROS deployment. 
- keypointnet_ros_msgs is messages used for keypoint_ros

## Overview  
## Citing  
## Installation, and Building (Compiling)  
##### 0. Install ROS and create your workspace   
##### 1. Establish your keypointnet python environment *keypointnet*, and virtual environment by conda is highly-recommended. (Install ROS at first, then install anaconda/miniconda)   
##### 2. Install the essential python packages and libraries according to [requirements.txt](requirements.txt) in your keypointnet env.   
##### 3. Fork this repository in */your_workspace_name/src/* by *$ git clone ...* or other method.   
##### 4. Compile ROS packages  
    $ cd ~/your_workspace_name/   
    $ conda activate your_env_name or $ source activate your_env_name    
    $ catkin_make  
    Or $ catkin_make -DPYTHON_EXECUTABLE=/home/dongyi/anaconda3/envs/paddle_env/bin/python 
    $ source devel/setup.bash
    $ ~/your_workspace_name/src/keypointnet_ros/keypointnet_ros/scripts/  
    
    $ mkdir trained_models
    $ mkdir dataset
     

## Before your Start
1. Modify the parameters in [keypointnet_ros/scripts/inference/config.py](keypointnet_ros/scripts/inference/config.py) in accordance with your configurations, including batchsize, learning rate, best model path, root of trainset, etc.  
2. Adjust the Shebang line (the first line in .py file, e.g. #!/usr/bin/python3) in KeypointDetector.py, keypoints_train.py, keypoints_test.py to your location of (virtual) python core.  
3. Adjust your data preprocessing and augmentation method by *... = trans.ComposeWithPoint...*  and *... = trans.Compose...* lines in [keypoints_train.py](keypointnet_ros/scripts/keypoints_train.py), [keypoints_pred.py](keypointnet_ros/scripts/inference/keypoints_pred.py) and [keypoints_test.py](keypointnet_ros/scripts/keypoints_test.py) according to [transforms.py](keypointnet_ros/scripts/transforms.py) and your data.  

## Training KeypointNet
##### 1. A jupyter notebook [2DShoesKeypointDetection_clear_version.ipynb](keypointnet_ros/scripts/2DShoesKeypointDetection_clear_version.ipynb) is provided for training and testing. Users can easily understand and train the model by following the instructions and code comments in it.  
##### 2. Run [keypoints_train.py](keypointnet_ros/scripts/keypoints_train.py) in linux shell or python interpreter 
    
    $ conda activate keypointnet (your_env_name) or $ source activate keypointnet (your_env_name)  
    $ cd ~/your_workspace_name/src/keypointnet_ros/keypointnet_ros/scripts
    $ python keypoints_train.py or $ ./keypoints_train.py

## Testing KeypointNet
##### 1. Use the jupyter notebook [2DShoesKeypointDetection_clear_version.ipynb](keypointnet_ros/scripts/2DShoesKeypointDetection_clear_version.ipynb) for testing.
##### 2. Run [keypoints_test.py](keypointnet_ros/scripts/keypoints_test.py) in linux shell or python interpreter like [Training](README.md#Training). 
##### 3. Output Keypoint Detection Results 

## Post Processing: Keypoint-based State Classification and Orientation Estimation
Check [2DShoesKeypointDetection_clear_version.ipynb](keypointnet_ros/scripts/2DShoesKeypointDetection_clear_version.ipynb) and [evaluation/](keypointnet_ros/scripts/evaluation/) for details.

## Deployment on ROS (-based Robot System)
[KeypointDetector.py](keypointnet_ros/scripts/KeypointDetector.py) works in ROS environment and achieves fucntions below by multi-threads  

### Node Initialization: keypointnet_ros  
Our main node.  

### Subscribing Topics  
Subscribing msgs in topic '/camera/color/image_raw', '/darknet_ros/bounding_boxes', '/darknet_ros/detection_image' 

### Inference: Keypoints Detection, Keypoing-based Pose Classfication and Orientation Estimation 
Inferring the **keypoints, states, orientations** of raw camera images by call functions in [keypoints_pred.py](keypointnet_ros/scripts/inference/keypoints_pred.py)     
Set the value of state_mode and orient_mode for the KPinfer function to infer the states and orientations according to your requirements.  

### Publishing Topics  
Publishing msgs in topic '/keypointnet_ros/state_keypoints', '/keypointnet_ros/keypoint_image'   

### Messages 
    from cv_bridge import CvBridge, CvBridgeError  
    from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox # , ObjectCount  
    from std_msgs.msg import String  
    from sensor_msgs.msg import Image  
    from keypointnet_ros_msgs.msg import Keypoint, Keypoints, KeyObjects  
where **Keypoint** is defined by [Keypoint.msg](keypointnet_ros_msgs/msg/Keypoint.msg) as    

    float64 confidence  
    int64 x  
    int64 y  
    string kp_class  
    
,**Keypoints** is defined by [Keypoints.msg](keypointnet_ros_msgs/msg/Keypoints.msg) as  

    Keypoint[] keypoints  
    string state  
    float64 alpha  
,**KeyObjects** is defined by [KeyObjects.msg](keypointnet_ros_msgs/msg/KeyObjects.msg) as   

    Keypoints[] objects

### Start to run the keypointnet_ros
    $ cd ~/your_workspace_name/   
    $ conda activate keypointnet (your_env_name) or $ source activate keypointnet (your_env_name)   
    $ catkin_make  
    #$ catkin_make -DPYTHON_EXECUTABLE=/home/dongyi/anaconda3/envs/paddle_env/bin/python 
    #$ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 
    
    $ source devel/setup.bash  
    
    #$ roslaunch realsense2_camera rs_camera.launch 
    $ roslaunch realsense2_camera rs_camera.launch align_depth:=true 
    
    $ roslaunch darknet_ros yolov4-p5RP.launch   
    Or $ roslaunch darknet_ros yolo_v3_tiny_spb.launch  
    
    # $ roscore # $ roslauch calls ros master automatically, thus no need to roscore again.   
    $ rosrun keypointnet_ros KeypointDetector.py  

### Call keypoint_ros in robot system (To Be Validated and Improved)     
    $ roslaunch realsense2_camera rs_camera.launch align_depth:=true  
    
    # run the calibration software  
    $ roslaunch easy_handeye eye_to_hand_calibration.launch  
    
    $ roslaunch darknet_ros yolov4-p5RP.launch
    $ rosrun keypointnet_ros KeypointDetector.py
    $ rosrun ur_smach visionInterface.py
    ...  
    
## Files Description  
1. [/keypointnet_ros/scripts/](keypointnet_ros/scripts/) includes python scripts   
    1. [/inference/config.py](keypointnet_ros/scripts/inference/config.py) is the configuration file of training and deployment our keypoint detection, which should be input and adjusted by users.  
    2. [KeypointDetector.py](keypointnet_ros/scripts/KeypointDetector.py) works in ROS environment and achieves fucntions below by multi-threads  
        - Subscribing msgs in topic '/camera/color/image_raw', '/darknet_ros/bounding_boxes', '/darknet_ros/detection_image'  
        - Inferring the keypoints, states, orientations of raw camera images by call functions in keypoints_pred.py  
        - Publishing msgs in topic '/keypointnet_ros/state_keypoints', '/keypointnet_ros/keypoint_image' 
    3. [keypoints_test.py](keypointnet_ros/scripts/keypoints_test.py) works in python environment and tests the keypoint detection performance by a small batch of images.
    4. [keypoints_train.py](keypointnet_ros/scripts/keypoints_train.py) works in python environment and trains the keypoint detection model according to config.py.
    5. [keypoints_pred.py](keypointnet_ros/scripts/inference/keypoints_pred.py) can accomplish the inference of keypoints and states in deployment and testing segment. It works for KeypointDetector.py anf keypoints_test.py.  
    6. . [/utils/KPDataset.py](keypointnet_ros/scripts/utils/KPDataset.py) and [/utils/PCDataset.py](keypointnet_ros/scripts/utils/PCDataset.py) read and augment date, then generate dataset classes to make them understandable for PaddlePaddle.  
    7. [/models/](keypointnet_ros/scripts/models/) and **/trained_models/** provide network (model) architecture and pre-trained weights respectively.  
    8. [transforms.py](keypointnet_ros/scripts/transforms.py) with [functional.py](keypointnet_ros/scripts/functional.py) contains many useful image data augmentation (transformation) methods, which were not only designed for this task, but also suitable for other image-based Machine Learning tasks (i.e., classification, segmentation, regression and keypoint detection). They are not framework-limited, and it means that any python-based framework like PyTorch, PaddlePaddle can use them.  
    9. [2DShoesKeypointDetection_clear_version.ipynb](keypointnet_ros/scripts/2DShoesKeypointDetection_clear_version.ipynb) is the jupyter notebook version of our code.  
    10. [dataset/](keypointnet_ros/scripts/dataset/) stores the dataset for training.  
2. [/keypointnet_ros_msgs/](keypointnet_ros_msgs/) involves ROS Messages - related code  
3. [/keypointnet_ros/src/](keypointnet_ros/scripts/keypointnet_ros/src/) consists of cpp code  

## Notes: Create your ROS packages from scratch
    # Create your workspace  
    $ cd ~/catkin_workspace/src
    $ catkin_create_pkg keypointnet_ros_msgs actionlib_msgs geometry_msgs sensor_msgs message_runtime std_msgs   
    $ catkin_create_pkg keypointnet_ros rospy roscpp std_msgs cv_bridge sensor_msgs darknet_ros_msgs keypointnet_ros_msgs image_transport message_generation message_runtime nodelet actionlib  
    $ cd ~/catkin_workspace  
    $ source devel/setup.bash 

## To Be Uptated
1. Keypoint-based states classification
2. Keypoint-based Pose Estimation

## To Do
11. Train resnet34 classification model without @paddle.jit.to_static to acuquire a model with smaller size    
12. Table. 2. To compare shoe keypoint detection algorithms (precision, sample efficiency (the size of the dataset), inference speed)?   

