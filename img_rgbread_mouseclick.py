# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:26:43 2020

@author: atakan
"""

import cv2
import numpy as np
import datetime

def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,'' , y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(x) + ',' + str(y)
        cv2.putText(img,strXY,(x,y),font,.5,(255,255,0),2)
        cv2.imshow('image',img)
    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y,x,0]
        green = img [y,x,1]
        red = img [y,x,2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        strBGR=str(blue) + ',' + str(green) + ','+ str(red)
        cv2.putText(img,strBGR,(x,y),font,.5,(0,255,255),2)
        cv2.imshow('image',img)
        
        
img=cv2.imread("data/lena.jpg",1)
cv2.imshow('image',img)
cv2.setMouseCallback('image',click_event)

cv2.waitKey(0)













    
