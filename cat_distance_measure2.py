# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:40:53 2021

@author: gat06
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt

def click_event(event,x,y,flags,param):
    global xt,yt
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),3,(0,0,255),-1)
        points.append((x,y))
        p00 =  -41.99
        p10 = 0.2985 
        p01 = 0.1087
        p20 = 1.411e-06
        p11 =  -0.0004286
        p02= 7.678e-05
        xt=round(p00 + p10*x + p01*y + p20*x**2 + p11*x*y+ p02*y**2,2)
        
        py00 =  -10.82 
        py10 = 0.008127 
        py01 = 0.6725
        py20 =  -9.269e-06 
        py11 =  1.81e-06
        py02 =  -0.0009003
        yt=round(py00 + py10*x + py01*y + py20*x**2 + py11*x*y+ py02*y**2,2)

        print(xt,yt)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY=str(xt) + ',' + str(yt)
        cv2.putText(img,strXY,(x,y),font,.5,(0,0,0),2)

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


