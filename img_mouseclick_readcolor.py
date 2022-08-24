# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:56:54 2020

@author: atakan
"""

import cv2
import numpy as np

def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        blue = img[y,x,0]
        green = img [y,x,1]
        red = img [y,x,2]
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        mycolorImage = np.zeros((512,512,3),np.uint8)
        mycolorImage[:]=[blue,green,red]
        cv2.imshow('color',mycolorImage)

#img=np.zeros((512,512,3),np.uint8)
img=cv2.imread("data/lena.jpg",1)
cv2.imshow('image',img)
points = []
cv2.setMouseCallback('image',click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()