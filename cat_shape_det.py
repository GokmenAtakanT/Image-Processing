# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:36:38 2021

@author: gat06
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(1)

ret, frame1 = cap.read()
frame1 = cv2.flip(frame1,1)
ret, frame2 = cap.read()
frame2 = cv2.flip(frame2,1)
x1=0
y1=0
x1_old=0
y1_old=0
center=(int(x1),int(y1))
#print(frame1.shape)

val_x=[0]
val_y=[0]

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (15,15), 0)
    #blur = cv2.medianBlur(blur, 15)
    
    _, thresh = cv2.threshold(blur, 15, 255, cv2.THRESH_BINARY)
    
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
       # x, y, _, _ = cv2.boundingRect(contour)
        
        (x1,y1),radius = cv2.minEnclosingCircle(contour)
        center = (int(x1),int(y1))
        radius = int(radius)
        
        val_x.append(x1)
        val_y.append(y1)
        
        print(val_x[-1], " ", val_y[-1])
        x1_old=x1
        y1_old=y1
    cv2.circle(frame1,center,5,(0,0,255),2)
    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    text = "x: " + str(round(val_x[-1])) + ", y: " + str(round(val_y[-1]))
    
    cv2.putText(frame1, text, center,
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #image = cv2.resize(frame1, (1280,720))
    cv2.imshow("feed", frame1)
    frame1 = frame2
    
    ret, frame2 = cap.read()
    frame2 = cv2.flip(frame2,1)

    
    #cnt = contours[0]
    #M = cv2.moments(cnt)
    #print( M )
    #print(contours[0][0])
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break
cv2.destroyAllWindows()
cap.release()







