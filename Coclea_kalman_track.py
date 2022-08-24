# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:11:16 2022

@author: atakan
"""


import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from time import time
previous = time()

p1_val=[]
p2_val=[]
angle_val=[]
centre_val=[]

cap = cv2.VideoCapture('coclea_close8.mp4')
#cap = cv2.VideoCapture(0);
i=0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_paper_i2.avi', fourcc, 20.0, (640,  480))

kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
kf.errorCovPre= np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
kf.errorCovPost= np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
x1_o=0
y1_o=0
x1_p=0
y1_p=0
data_val=[]
center = (0,0)
while cap.isOpened():
    current = time()- previous
    ret, img= cap.read()
    img=cv2.flip(img, 0)
    img=cv2.flip(img, 1)
    diff=img
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ret==False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    l_b = np.array([60, 0, 0])
    u_b = np.array([255, 255, 60])
    mask = cv2.inRange(hsv, l_b, u_b)
    mask = cv2.equalizeHist(mask)
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(3, 3))
    # mask= clahe.apply(mask)
    #mask = cv2.GaussianBlur( mask, (3,3), 0)
    #mask = cv2.medianBlur(mask, 3)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=6)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=9) 
    #ret, thresh = cv2.threshold(imgray, 0, 50, 0)

    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    areas = [cv2.contourArea(c) for c in contours]
    if  len(areas)>0:
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        # Calculate the area of each contour
        area = cv2.contourArea(cnt)
        # Ignore contours that are too small or too large
        (x_m,y_m),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x_m),int(y_m))
        radius = int(radius)
        #cv2.circle(img,center,radius,(0,255,0),2)
        
        measured = np.array([[np.float32(int(x_m))], [np.float32(int(y_m))]])
        kf.correct(measured)
        (predicted) = kf.predict()
        x1=int(predicted[0])
        x2=int(predicted[1])
        x1_p=x_m+(x1-x_m)+(x1_p-x1_o)*0.01
        y1_p=y_m+(x2-y_m)+(y1_p-y1_o)*0.01
        x1_o=x1_p
        y1_o=y1_p
    cv2.circle(img,center,1,(255,0,0),2)
    xt=round((0.03422*x1_p-13.24),2)
    yt=round((0.03279*y1_p-7.71),2)
    value=(xt,yt)
    cv2.circle(img,(int(x1_p),int(y1_p)),1,(0,0,255),2)
    cv2.circle(img,(int(x1_p),int(y1_p)),20,(0,255,0),2)
    cv2.putText(img, str(value), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    data_val.append(value)
            #cv2.drawContours(img, contours, i, (0, 255, 0), 1)
    cv2.imshow('Image', img)
    cv2.imshow('mask',mask)
    #out.write(img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # if ret==False:
    #     print("Can't receive frame (stream end?). Exiting ...")
    # break
cap.release()
#out.release()
cv2.destroyAllWindows()