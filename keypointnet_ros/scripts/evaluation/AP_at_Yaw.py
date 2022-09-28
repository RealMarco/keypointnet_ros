#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:33:52 2022

@author: marco
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from io import StringIO

matplotlib.rcParams['font.family']="Arial" # "Comic Sans MS". Change default font 

plt.figure(1)
#font = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
#plt.plot([0,29],[0.90,0.90],color='k',linestyle='--',linewidth=0.4)
#plt.plot([0,29],[0.80,0.80],color='k',linestyle='--',linewidth=0.4)

#net0 = pd.read_csv(r'F:\OD and GD Based on DL\Results Samples\run-2nd_OW_noBN-tag-loss_IOU.csv', usecols=['Step', 'Value'])
#thresh_KP_overall = [0,1,2,3,4,5,10]
#AP_KP_overall = [0, 0.5474, 0.8526,0.9632,0.9842,0.9947,1]
thresh_KP = [0,1,2,3,4,5,6,7,8,9,10]

AP_KP_bottom = [0,0.5357,0.7143,0.8929,0.9286,0.9643,0.9643,1,1,1,1]
plt.plot(thresh_KP, AP_KP_bottom, lw=2, label='Our Method - bottom', color='#82B0D2') # ,linestyle='--'
avg_yaw_error_bottom = 1.3778
plt.plot([avg_yaw_error_bottom ,avg_yaw_error_bottom],[0,1], color='#82B0D2',linestyle='--',linewidth=1)

AP_KP_side = [0,0.5753,0.8904,1,1,1,1,1,1,1,1]
plt.plot(thresh_KP, AP_KP_side, lw=2, label='Our Method - side', color='#BEB8DC') # ,linestyle='--'
avg_yaw_error_side = 0.9769
plt.plot([avg_yaw_error_side ,avg_yaw_error_side],[0,1], color='#BEB8DC',linestyle='--',linewidth=1)

AP_KP_top = [0,0.5281,0.8652,0.9551,0.9888,1,1,1,1,1,1]
plt.plot(thresh_KP, AP_KP_top, lw=2, label='Our Method - top', color='#FA7F6F') # ,linestyle='--'
avg_yaw_error_top = 1.0921
plt.plot([avg_yaw_error_top ,avg_yaw_error_top],[0,1], color='#FA7F6F',linestyle='--',linewidth=1)

thresh_CenterPose_top = [0,1,2,3,4,5,6,7,8,9,10]
AP_CenterPose_top = [0, 0.3373, 0.4819, 0.6506, 0.7711, 0.8434,0.8916,0.9277,0.9398, 0.9398, 0.9398]
plt.plot(thresh_CenterPose_top, AP_CenterPose_top, lw=2, label='CenterPose (ICRA 2022) - top', color='#999999')
avg_yaw_error_CP= 6.2766
plt.plot([avg_yaw_error_CP ,avg_yaw_error_CP],[0,1], color='#999999',linestyle='--',linewidth=1)


title1= 'Average Precision at Yaw Error'
plt.title(title1)
plt.legend(loc='lower right', fontsize =7)
plt.xlim((0,10))
plt.ylim((0,1.02))
plt.xlabel('Yaw Error (degrees)') #fontproperties=font
plt.ylabel('Average Precision')
plt.savefig('%s.png' % (title1), dpi=300)
plt.show()
