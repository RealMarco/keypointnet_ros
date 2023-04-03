#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
"""
@author: marco
"""
'''
Class Keypoints
Keypoint[] keypoints
string state
float64 alpha

Class Keypoint
float64 confidence
int64 x
int64 y
string kp_class
'''

#import os
#import threading
import sys
sys.path.append('/home/marco/robotic_sorting/src/keypointnet_ros/keypointnet_ros/scripts')
# sys.path.append('/home/dongyi/anaconda3/envs/paddle_env/lib/python3.9/site-packages')
if '/usr/lib/python3/dist-packages' in sys.path: # before importing other modules or packages
    sys.path.remove('/usr/lib/python3/dist-packages')
print (sys.path)

# sys.path.remove('/usr/lib/python3/dist-packages')
# sys.path.remove('/opt/ros/noetic/lib/python3/dist-packages')

import rospy 
import numpy as np 
# import sys
#path.sys.append('/home/dongyi/anaconda3/envs/paddle_env/lib/python3.9/site-packages')
from threading import Thread

import cv2
from cv_bridge import CvBridge, CvBridgeError
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox # , ObjectCount
from std_msgs.msg import String
from sensor_msgs.msg import Image # ?
from keypointnet_ros_msgs.msg import Keypoint, Keypoints, KeyObjects

from models.resnet34_classification_paddle import Model_resnet34
from models.keypointnet_deepest_paddle import KeypointNet_Deepest 
from inference.keypoints_pred import KPinfer, PCinfer, get_trained_model
from inference.config import best_PCmodel_path, best_PCmodel_path2, best_KPmodel_path

# References:
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
# https://gitlab.com/neutron-nuaa/robot-arm/-/tree/main/Paper_manipulation_and_shoe_packaging/darknet_ros
# https://www.paddlepaddle.org.cn/documentation/docs
# http://wiki.ros.org/rospy
# http://wiki.ros.org/rospy_tutorials/Tutorials/
# https://docs.ros.org/en/melodic/api/rospy/html/

# define a camera image callback function
def cimgCallback(cimg):
#    bridge =  CvBridge()
    try:
        cv_cimg = cvbridge.imgmsg_to_cv2(cimg, "rgb8") # "bgr8", desired_encoding="passthrough"
    except CvBridgeError as e:
        rospy.logerr('Converting camera image error: ' + str(e))
    global OriImage # essential to claim a variable as global 
    OriImage = cv_cimg
    rospy.loginfo("Subscribing an image from camera with a shape of (%d, %d, %d) "%(OriImage.shape[0],OriImage.shape[1],OriImage.shape[2])) # (H,W,C)
    
# define an raw image subscriber
def camera_img_subscriber():
#    rospy.init_node('camera_img_subscriber', anonymous=True)
    rospy.Subscriber('/camera/color/image_raw', Image, cimgCallback, queue_size=1) # '/camera/image_raw'
#    rospy.spin()

# define an infer callback function, which would cropping OriImage by bbxes firtly
def inferCallback(bbxes):
    #rospy.sleep(0.2)
    global ShoeBbxes, CroppedImgs, CroppedXYmin, CroppedXYmax # not required to claim a list as global
    ShoeBbxes.clear() # clear the existing ShoeBbxes; global ShoeBbxes ShoeBbxes=[]
    for bbx in bbxes.bounding_boxes:
        if bbx.Class == 'shoe':  # 
            [xmin, ymin, xmax, ymax] = [round(bbx.xmin), round(bbx.ymin), round(bbx.xmax), round(bbx.ymax)]
            ShoeBbxes.append([xmin, ymin, xmax, ymax])
            rospy.loginfo("Subscribing a shoe bounding box: xmin:%d ymin:%d xmax:%d ymax:%d"
                  %(xmin, ymin, xmax, ymax))
    
    if ShoeBbxes != []: # avoid keypoint inference and publishment when no shoe was detected
        # crop original image by bounding boxes
        CroppedImgs.clear() # clear the existing CroppedImgs
        CroppedXYmin.clear()
        CroppedXYmax.clear()
        for ShoeBbx in ShoeBbxes:
    #        global OriImage, CroppedImgs
            cropped_img = OriImage[ShoeBbx[1]:ShoeBbx[3], ShoeBbx[0]:ShoeBbx[2]]# [ymin:ymax,xmin:xmax]
            CroppedImgs.append(cropped_img)
            CroppedXYmin.append([ShoeBbx[0], ShoeBbx[1]])
            CroppedXYmax.append([ShoeBbx[2], ShoeBbx[3]])
            
            
        # Infer CroppedImgs list and return keypoints + shoe pose/states classes.
        global state_classes, confident_kps, orientations, states
        global PCmodel, PCmodel2, KPmodel
        ### May use multi threads to speed up the inference
        ### Infer State class
        
        state_classes =PCinfer(PCmodel,PCmodel2, CroppedImgs) # CroppeedImg num x 1
        
        ### Infer keypoints with confidence in the format of [toe_c, toe_x, toe_y, heel_c ...,  inside ..., outside ..., topline ... ] 
        confident_kps, orientations, states = KPinfer(KPmodel,CroppedImgs,orient_mode =True,state_mode = True)  #the size of confident_kps is CroppeedImg num x 15, including coordinates in image coordinate system
        
        # publish keypoints with state, orientation and its KPImage
        #rospy.sleep(0.2)
        keypoint_publisher()

# define a BoundingBoxes subscriber
def bbxes_subscriber():
    # initialize ros node 
#    rospy.init_node('bbxes_subscriber', anonymous=True)
    
    # Registration: create a subscriber, and subscribing a topic named bounding_boxes with BoundingBoxes message
    # Register bbxes Callback function
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, inferCallback, queue_size=1) # 'bounding_boxes'
    
    # recurrently subscribe the bbxes
#    rospy.spin() # blocking function

#define an object detection image (marked with bouding boxes) callback function
def odImgCallback(odImg):
#    bridge =  CvBridge()
    try:
        cv_odImg = cvbridge.imgmsg_to_cv2(odImg, "rgb8") # bgr8, desired_encoding="passthrough"
    except CvBridgeError as e:
        rospy.logerr('Converting object detection image error: ' + str(e))
    global ODImage # essential to claim a variable as global 
    ODImage = cv_odImg
    rospy.loginfo("Subscribing an object detection image with a shape of (%d, %d, %d)"%(ODImage.shape[0],ODImage.shape[1], ODImage.shape[2])) # (H,W,C)

# define an object detection image subscriber
def od_img_subscriber():
#    rospy.init_node('bbx_img_subscriber', anonymous=True)
    rospy.Subscriber('/darknet_ros/detection_image', Image, odImgCallback, queue_size=1) # '/camera/image_raw'
#    rospy.spin()

    
# define a keypoints, shoe state publisher, orientation publisher, publish 
def keypoint_publisher():
    # init ros node
#    rospy.init_node('shoe_state_publiser', anonymous=True) # try anonymous person
    try:
        # rospy.sleep(0.1) # wait for finishing node registration, or the 1st msg wouldn't be published
        # if len(state_classes)==len(confident_kps):
        # create msg
        global ShoeBbxes, KPImage, state_classes, confident_kps, orientations, states, OriImage, CroppedXYmin, CroppedXYmax #ODImage
        KPImage = OriImage #ODImage
        KeyShoes = KeyObjects()  # Keypoints[] objects
        for i in range(len(confident_kps)): # number of shoeBBxes or CroppeedImgs
            kp_state =  Keypoints() # keypoints with state
            #kp_state.state = shoe_states[states[i]]  # for keypoint-based classification
            kp_state.state = shoe_states[state_classes[i]] # for direct state classification
            kp_state.alpha = orientations[i][0]
            
            c_list = []
            for j in range(confident_kps.shape[1]//3):
                c_list.append(confident_kps[i][3*j])
            if kp_state.state == 'side': 
                minc_j = c_list.index(min(c_list[2:4])) # the j index of minima keypoint confidence
            elif kp_state.state == 'bottom':
                minc_j = 4
            else: # top
                minc_j = 6 # j index out of range to avoid filter
                
            # kp_state.keypoints ...
            for j in range(confident_kps.shape[1]//3):
                #if confident_kps[i][3*j]>0.5: # confidence threshold
                kp =Keypoint()
                if j != minc_j: # output keypoints according to the state
            	    kp.confidence = confident_kps[i][3*j]
    	            # transfer from cropped image coordinate system to original image coordinate system
    	            kp.x = round(confident_kps[i][3*j+1] + CroppedXYmin[i][0]) # ShoeBbxes[i][0]) + 0.5 pixel coordinate, no need to +0.5
    	            kp.y = round(confident_kps[i][3*j+2] + CroppedXYmin[i][1]) # ShoeBbxes[i][1])#
    	            kp.kp_class = keypoint_classes[j]
    	        
    	            kp_state.keypoints.append(kp)
    	        
    	            #Visualize keypoints
    	            KPImage =cv2.circle(KPImage, (kp.x, kp.y), 10, kp_colors[j], -1) # circle(img, point_center, radius, RGB, thickness
    	            KPImage=cv2.putText(KPImage, kp.kp_class+str(round(kp.confidence,2)), (kp.x+6, kp.y-6), cv2.FONT_HERSHEY_SIMPLEX , 0.5, kp_colors[j], 1, cv2.LINE_AA) # kp.kp_class keypoint_classes[j]
                    
            #Visualize state
            KPImage=cv2.rectangle(KPImage, (CroppedXYmin[i][0], CroppedXYmin[i][1]), (CroppedXYmax[i][0],CroppedXYmax[i][1]), (0,255,0),2) # (img,(left, top),(right, bottom), color, thickness)
            KPImage=cv2.putText(KPImage, "shoe DL-%s KP-%s"%(shoe_states[state_classes[i]], kp_state.state), (CroppedXYmin[i][0],CroppedXYmin[i][1]-24), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0,255,0), 2, cv2.LINE_AA)
            #KPImage=cv2.putText(KPImage, "shoe KP-%s"%kp_state.state, (CroppedXYmin[i][0],CroppedXYmin[i][1]-24), cv2.FONT_HERSHEY_SIMPLEX , 0.6, (0,255,0), 2, cv2.LINE_AA) 
            KPImage=cv2.putText(KPImage, 'yaw = %.2f'%kp_state.alpha, (CroppedXYmin[i][0],CroppedXYmin[i][1]-8), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,255,0), 1, cv2.LINE_AA) # ord('°')=176, chr(176)='°'
            # putText(Image, text, bottom-left corner, font, fontScale, color, thickness, lineType, bottomLeftOrigin)
            # publish keypoints with state 
            
            KeyShoes.objects.append(kp_state)
            #kp_state_publisher.publish(kp_state)
            #rospy.loginfo("Publishing shoe keypoints with state %s"%(kp_state.state))
        # publish keypoints of shoes
        kp_state_publisher.publish(KeyShoes)
        rospy.loginfo("Publishing the keypoints of %d shoes"%(len(KeyShoes.objects)))
        
        # publisher Present the keypoints and shoe class on ODImage, similar to the topic /darknet_ros/detection_image in object detection
        # KPImage =  np.asarray(KPImage)
        cv2.imshow("Keypoint Detection", KPImage[:, :, ::-1]) # RGB to BGR
        while (cv2.waitKey(30)==27):
            pass
        KPImage_msg = cvbridge.cv2_to_imgmsg(KPImage, "bgr8") #"bgr8", encoding="passthrough"
        kp_img_publisher.publish(KPImage_msg)
        rospy.loginfo("Publishing an keypoint detection image with a shape of (%d, %d, %d)"%(KPImage.shape[0],KPImage.shape[1],KPImage.shape[2])) 
    except rospy.ROSInterruptException: # except [error type]
        pass 

# define a keypoints (in original image coordinate system), shoe class (top side bottom) () subscriber


if __name__ == "__main__":# avoid automatic running below lines when this .py file is imported by others.
#    try:
    OriImage = np.zeros((2,2,3))
    # ODImage  = np.zeros((2,2)) 
    KPImage =  np.ones((2,2,3))
    ShoeBbxes = []
    CroppedImgs = []
    CroppedXYmin = []  # record the [[xmin, ymin],] for coordinate transferring 
    CroppedXYmax = []
    state_classes = [] # direct state classification
    states = [] # keypoint-based state classification
    confident_kps= []
    orientations = []
    shoe_states= ['top','side','bottom']
    keypoint_classes =  ['toe','heel','inside','outside','topline']
    # kp_colors = [(136,32,29), (0,0,192), (160,48,112),(171,171,175),(233,44,242)] # RGB
    kp_colors = [(136,32,29), (0,0,192), (139,69,19),(136,136,136),(233,44,242)] # RGB
    cvbridge = CvBridge()
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
    
    rospy.init_node("keypointnet_ros", anonymous=True)
    #rospy.init_node('keypointnet_ros')
    # Registration: Create a publisher, and publish a topic named person_info with test_topic.msg.Person message, queue size =4
    kp_state_publisher= rospy.Publisher('/keypointnet_ros/state_keypoints', KeyObjects, queue_size=1) # latch =
    kp_img_publisher  = rospy.Publisher('/keypointnet_ros/keypoint_image', Image, queue_size=1)
    rospy.sleep(0.2) # wait for finishing node registration, or the 1st msg wouldn't be published
    
    # keep subscribing... cyclically
    t1 = Thread(target=camera_img_subscriber) # args=(arg_for_target_function,)
    t2 = Thread(target=bbxes_subscriber)  # subscribe bbxes and infer keypoinis + state class, then publish Keypoints, KPImage
    #t3 = Thread(target=od_img_subscriber)
    t1.start()
    rospy.sleep(0.2) # wait for camera_img_subscriber
    #t3.start()
    t2.start()
    
#        # lock CroppedImgs, OriImage, ODImage when t2 is running
#        .join()
#        lock = threading.Lock()
    rospy.spin()
    
    # Recurrently do publishing or publish in inferCallback
        
#    except rospy.ROSInterruptException: # except [error type]
#        pass 
    
    
