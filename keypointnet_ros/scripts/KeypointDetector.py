#!/usr/bin/python3
# -*- coding: utf-8 -*-
# 
"""
@author: marco
"""
'''
Class BoundingBox
float64 probability
int64 xmin
int64 ymin
int64 xmax
int64 ymax
int16 id
string Class

Class BoundingBoxes
Header header
Header image_header
BoundingBox[] bounding_boxes
'''
'''
Class Keypoints
Keypoint[] keypoints
string state

Class Keypoint
float64 confidence
int64 x
int64 y
string kp_class
'''

#import os
#import threading
from threading import Thread

import cv2
import numpy as np 

import rospy 
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox # , ObjectCount
from std_msgs.msg import String
from sensor_msgs.msg import Image # ?
from keypointnet_ros_msgs.msg import Keypoints, Keypoint

from keypoints_pred import get_trained_model, KPinfer, PCinfer, best_PCmodel_path, best_PCmodel_path2, best_KPmodel_path
from models.resnet34_classification_paddle import Model_resnet34
from models.keypointnet_deepest_paddle import KeypointNet_Deepest # GResNet-Deepest

# References:
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
# https://gitlab.com/neutron-nuaa/robot-arm/-/tree/main/Paper_manipulation_and_shoe_packaging/darknet_ros
# https://www.paddlepaddle.org.cn/documentation/docs
# http://wiki.ros.org/rospy
# http://wiki.ros.org/rospy_tutorials/Tutorials/

# define a camera image callback function
def cimgCallback(cimg):
#    bridge =  CvBridge()
    try:
        cv_cimg = CvBridge.imgmsg_to_cv2(cimg, "bgr8") # desired_encoding="passthrough"
#    cimg = CvBridge.cv2_to_imgmsg(cv_cimg, "bgr8") # encoding="passthrough"
    except CvBridgeError as e:
        rospy.logerr('Converting camera image error: ' + str(e))
    global OriImage # essential to claim a variable as global 
    OriImage = cv_cimg
    rospy.loginfo("Subscribing an image from camera with a shape of ", OriImage.shape) # (H,W,C)
    
# define an raw image subscriber
def camera_img_subscriber():
#    rospy.init_node('camera_img_subscriber', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, cimgCallback) # '/camera/image_raw'
#    rospy.spin()

# define an infer callback function, which would cropping OriImage by bbxes firtly
def inferCallback(bbxes):
    rospy.sleep(0.4)
    # global ShoeBbxes # not required to claim a list as global
    ShoeBbxes.clear() # clear the existing ShoeBbxes; global ShoeBbxes ShoeBbxes=[]
    for bbx in bbxes.bounding_boxes:
        if bbx.Class == 'shoe':  # 
            [xmin, ymin, xmax, ymax] = [round(bbx.xmin), round(bbx.ymin), round(bbx.xmax), round(bbx.ymax)]
            ShoeBbxes.append([xmin, ymin, xmax, ymax])
            rospy.loginfo("Subscribing a shoe bounding box: xmin:%d ymin:%d xmax:%d ymax:%d"
                  %(xmin, ymin, xmax, ymax))
    
    # crop original image by bounding boxes
    CroppedImgs.clear() # clear the existing CroppedImgs
    CroppedXYmin.clear()
    for ShoeBbx in ShoeBbxes:
#        global OriImage, CroppedImgs
        cropped_img = OriImage[ShoeBbx[1]:ShoeBbx[3], ShoeBbx[0]:ShoeBbx[2]]# [ymin:ymax,xmin:xmax]
        CroppedImgs.append(cropped_img)
        CroppedXYmin.append([ShoeBbx[0], ShoeBbx[1]])
        
        
    # Infer CroppedImgs list and return keypoints + shoe pose/states classes.
    global state_classes
    global confident_kps
    global PCmodel, PCmodel2, KPmodel
    ### May use multi threads to speed up the inference
    ### Infer State class
    state_classes =PCinfer(PCmodel,PCmodel2, CroppedImgs) # CroppeedImg num x 1
    
    ### Infer keypoints with confidence in the format of [toe_c, toe_x, toe_y, heel_c ...,  inside ..., outside ..., topline ... ] 
    confident_kps= KPinfer(KPmodel,CroppedImgs)  #the size of confident_kps is CroppeedImg num x 15
    
    # publish keypoints with state and its KPImage
    keypoint_publisher()

# define a BoundingBoxes subscriber
def bbxes_subscriber():
    # initialize ros node 
#    rospy.init_node('bbxes_subscriber', anonymous=True)
    
    # Registration: create a subscriber, and subscribing a topic named bounding_boxes with BoundingBoxes message
    # Register bbxes Callback function
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, inferCallback) # 'bounding_boxes'
    
    # recurrently subscribe the bbxes
#    rospy.spin() # blocking function

#define an object detection image (marked with bouding boxes) callback function
def odImgCallback(odImg):
#    bridge =  CvBridge()
    try:
        cv_odImg = CvBridge.imgmsg_to_cv2(odImg, "bgr8") # desired_encoding="passthrough"
#    cimg = CvBridge.cv2_to_imgmsg(cv_cimg, "bgr8") # encoding="passthrough"
    except CvBridgeError as e:
        rospy.logerr('Converting object detection image error: ' + str(e))
    global ODImage # essential to claim a variable as global 
    ODImage = cv_odImg
    rospy.loginfo("Subscribing an object detection image with a shape of ", ODImage.shape) # (H,W,C)

# define an object detection image subscriber
def od_img_subscriber():
#    rospy.init_node('bbx_img_subscriber', anonymous=True)
    rospy.Subscriber('/darknet_ros/detection_image', Image, odImgCallback) # '/camera/image_raw'
#    rospy.spin()

    
# define a keypoints, shoe class publisher, State (orientation) publisher, publish 
def keypoint_publisher():
    # init ros node
#    rospy.init_node('shoe_state_publiser', anonymous=True) # try anonymous person
    try:
        rospy.sleep(0.2) # wait for finishing node registration, or the 1st msg wouldn't be published
        # if len(state_classes)==len(confident_kps):
        # create msg
        global KPImage, state_classes, confident_kps
        KPImage = ODImage
        for i in range(len(state_classes)): # number of shoeBBxes or CroppeedImgs
            kp_state =  Keypoints() # keypoints with state
            kp_state.state = shoe_states[state_classes[i]]
            kp =Keypoint()
            # kp_state.keypoints ...
            for j in range(confident_kps.shape[1]//3):
                kp.confidence = confident_kps[i][3*j]
                # transfer from cropped image coordinate system to original image coordinate system
                kp.x = confident_kps[i][3*j+1] + CroppedXYmin[i][0] # + 0.5 pixel coordinate, no need to +0.5
                kp.y = confident_kps[i][3*j+2] + CroppedXYmin[i][1]
                kp.kp_class = keypoint_classes[j]
                
                kp_state.keypoints.append(kp)
                
                #Visualize keypoints
                KPImage =cv2.circle(KPImage, (kp.x, kp.y), 8, kp_colors[j], -1) # circle(img, point_center, radius, BGR, thickness
            #Visualize state
            KPImage=cv2.putText(KPImage, kp_state.state, (CroppedXYmin[i][0], CroppedXYmin[i][1]), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 2, cv2.LINE_AA) 
            # putText(Image, text, bottom-left corner, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
            # publish keypoints with state 
            kp_state_publisher.publish(kp_state)
            rospy.loginfo("Publishing shoe keypoints with state %s"%(kp_state.state))
        
        # publisher Present the keypoints and shoe class on ODImage, similar to the topic /darknet_ros/detection_image in object detection
        KPImage_msg = CvBridge.cv2_to_imgmsg(KPImage, "bgr8") # encoding="passthrough"
        kp_img_publisher.publish(KPImage_msg)
        rospy.loginfo("Publishing an keypoint detection image with a shape of", KPImage.shape) 
    except rospy.ROSInterruptException: # except [error type]
        pass 

# define a keypoints (in original image coordinate system), shoe class (top side bottom) () subscriber


if __name__ == "__main__":# avoid automatic running below lines when this .py file is imported by others.
#    try:
    OriImage = np.zeros((4,4))
    ODImage  = np.zeros((2,2))
    KPImage =  np.ones((2,2))
    ShoeBbxes = []
    CroppedImgs = []
    CroppedXYmin = []  # record the [[xmin, ymin],] for coordinate transferring  
    state_classes = []
    confident_kps= []
    shoe_states= ['top','side','bottom']
    keypoint_classes =  ['toe','heel','inside','outside','topline']
    kp_colors = [(136,32,29), (0,0,192), (160,48,112),(171,171,175),(233,44,242)] # BGR
#        test_i = Image()
#        #test_i.
    
    # load models as global variables, then call them recurrently to predict the keypoints and shoe states
    # load the DL models then wait for cropped images to detect
    ### May use multi threads to speed up the loading
    ### Loading keypoint detection model v2
    KPmodel= KeypointNet_Deepest(input_channels=3, output_channels=5, channel_size=32,dropout=True, prob=0)
    KPmodel= get_trained_model(KPmodel, best_KPmodel_path)
    
    ### Loading Direct State Classification model v2
    PCmodel = Model_resnet34()
    PCmodel = get_trained_model(PCmodel, best_PCmodel_path)
    if best_PCmodel_path2 == '' or best_PCmodel_path2 == None:
        PCmodel2= None
    else:
        PCmodel2= Model_resnet34()
        PCmodel2= get_trained_model(PCmodel2,best_PCmodel_path2)
    
    rospy.init_node('keypointnet_ros', anonymous=True)
    # Registration: Create a publisher, and publish a topic named person_info with test_topic.msg.Person message, queue size =4
    kp_state_publisher= rospy.Publisher('/keypointnet_ros/state_keypoints', Keypoints, queue_size=2) # latch =
    kp_img_publisher  = rospy.Publisher('/keypointnet_ros/keypoint_image', Image, queue_size=1)
    
    # keep subscribing... cyclically
    t1 = Thread(target=camera_img_subscriber) # args=(arg_for_target_function,)
    t2 = Thread(target=bbxes_subscriber)  # subscribe bbxes and infer keypoinis + state class, then publish Keypoints, KPImage
    t3 = Thread(target=od_img_subscriber)
    t1.start()
    t3.start()
    t2.start()
    
#        # lock CroppedImgs, OriImage, ODImage when t2 is running
#        .join()
#        lock = threading.Lock()
    rospy.spin()
    
    # Recurrently do publishing or publish in inferCallback
        
#    except rospy.ROSInterruptException: # except [error type]
#        pass 
    
    
