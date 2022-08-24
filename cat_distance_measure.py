# -*- coding: utf-8 -*-
"""
Created on Mon May 24 14:52:13 2021

@author: gat06
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt

def click_event(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        points.append((x,y))
        print(x,y)
        cv2.imshow('image',img)


#img=np.zeros((512,512,3),np.uint8)
cap = cv2.VideoCapture(1)

ret, img = cap.read()
cv2.imshow('image',img)
points_x = []
points_y = []
points =[]
points_d=[]
#points=np.array(points)

cv2.setMouseCallback('image',click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()
