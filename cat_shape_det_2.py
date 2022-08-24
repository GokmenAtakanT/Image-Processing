# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 22:33:57 2021

@author: gat06
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:36:38 2021

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

cap = cv2.VideoCapture(1)

ret, frame1 = cap.read()
#frame1 = cv2.flip(frame1,1)
ret, frame2 = cap.read()
#frame2 = cv2.flip(frame2,1)
x=0
y=0
x1_old=0
y1_old=0
center=(int(x),int(y))
#print(frame1.shape)

val_x=[0]
val_y=[0]

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    blur = cv2.medianBlur(blur, 11)
    
    _, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
    
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours= cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) > 0 :
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		    c = max(contours, key=cv2.contourArea)
		    ((x, y), radius) = cv2.minEnclosingCircle(c)
		    M = cv2.moments(c)
		    center = (int(x), int(y))

    cv2.circle(frame1,center,5,(0,0,255),2)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    text = "x: " + str(x) + ", y: " + str(y)
    
    cv2.putText(frame1, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #image = cv2.resize(frame1, (1280,720))
    cv2.imshow("feed", frame1)
    frame1 = frame2
    
    ret, frame2 = cap.read()
    #frame2 = cv2.flip(frame2, 1)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cv2.destroyAllWindows()
cap.release()







