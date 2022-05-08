# KeypointNet: Keypoints Detection, States Classfication, Pose Calculation    
- keypointnet_ros consists of core code for Keypoints Detection, States Classfication and their corresponding ros packages.  
- keypointnet_ros_msgs is messages used for keypoint_ros  

## Files Description  
1. /keypointnet_ros/scripts/ includes python scripts  
 1) KeypointDetector.py achieves fucntions below by multi-threads  
  - Subscribing msgs in topic '/camera/color/image_raw', '/darknet_ros/bounding_boxes', '/darknet_ros/detection_image'  
  - Inferring the keypoints, states, (poses) of raw camera images by call functions in keypoints_pred.py  
  - Publishing msgs in topic '/keypointnet_ros/state_keypoints', '/keypointnet_ros/keypoint_image'  
 2) keypoints_pred.py can accomplish the inference of keypoints and states in deployment and testing segment.  
 3) /inference/config.py is the configuration file of DL models, which should be input and adjusted by users.  
 4) /utils/KPDataset.py and /utils/PCDataset.py read and augment date, then generate dataset classes to make them understandable for PaddlePaddle.  
 5) /models/ and /trained_models/ provide network (model) architecture and pre-trained weights respectively.  
 6) transforms.py with functional.py contains many useful image data augmentation (transformation) methods, which were not only designed for this task, but also suitable for other image-based Machine Learning tasks (i.e., classification, segmentation, regression and keypoint detection). They are not framework-limited, and it means that any python-based framework like PyTorch, PaddlePaddle can use them.  

2. /keypointnet_ros/src/ includes cpp code

## To Be Uptated
1. Pose Estimation
2. Keypoint-based states classification
3. Threshold or state based keypoint filter.

## To Do
4. Deploy YOLOv4-p5 on darknet_ros. Changes are required in the darknet_ros besides replacing darknet(v3) by darknetv4
 1) Change cofig/yolov3-tiny-spb.yaml file 
 2) Change cfg and .weights in yolo_network _config 
 3) launch/yolo_v3.lauch  
 4) Is it essential to adjust the files in darknet_ros/src? 
 5) Is it essential to adjust the /launch/darknet_ros.lauch? 
 6) recompile 
5. What about using KeypointNet to train and infer the states.
6. Train resnet34 classification model without @paddle.jit.to_static to acuquire a model with smaller size
