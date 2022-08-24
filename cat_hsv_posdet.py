# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 11:21:03 2021

@author: gat06
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
greenLower = (40, 20, 0)
greenUpper = (90, 255, 140)

cap = cv2.VideoCapture(1)

ret, frame = cap.read()
#frame1 = cv2.flip(frame1,1)
#frame2 = cv2.flip(frame2,1)
x=0
y=0
center=(int(x),int(y))
#print(frame1.shape)
val_x=[0]
val_y=[0]

while cap.isOpened():
    ret, frame = cap.read()
    blurred  = cv2.medianBlur(frame, 5)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)    
    mask = cv2.erode(mask, None, iterations=3)
    mask = cv2.dilate(mask, None, iterations=6)    
    contours= cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) > 0 :
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		    c = max(contours, key=cv2.contourArea)
		    ((x, y), radius) = cv2.minEnclosingCircle(c)
		    M = cv2.moments(c)
		    center = (int(x), int(y))

    cv2.circle(frame,center,5,(0,0,255),2)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    text = "x: " + str(x) + ", y: " + str(y)
    
    cv2.putText(frame, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #image = cv2.resize(frame1, (1280,720))
    cv2.imshow("feed", frame)
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cv2.destroyAllWindows()
cap.release()







