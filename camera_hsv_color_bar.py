# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:11:31 2021

@author: gat06
"""

import cv2
import numpy as np
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def nothing(x):
    pass

#cap = cv2.VideoCapture(0);
cap = cv2.VideoCapture('coclea_close8.mp4')
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 60, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 50, 255, nothing)
i=0

res=0
while(cap.isOpened()):
    ret, frame = cap.read()
    diff=frame

    #blurred  = cv2.medianBlur(frame, 5)
    #blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    #
    if ret==False:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_b, u_b)
    
    mask = cv2.equalizeHist(mask)
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.GaussianBlur( mask, (3,3), 0)
    #mask = fgbg.apply(mask)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=6)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=9) 
    #cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        # Draw each contour only for visualisation purposes
        cv2.drawContours(frame, contours, i, (0, 255, 0), 1)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    if cv2.waitKey(1) & 0xFF == 27 :
        break

cap.release()
cv2.destroyAllWindows()