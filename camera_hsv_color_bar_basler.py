# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:11:31 2021

@author: gat06
"""

import cv2
import numpy as np
from pypylon import pylon

#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def nothing(x):
    pass

#cap = cv2.VideoCapture(0);
cap = cv2.VideoCapture('coclea_close8.mp4')
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 100, 255, nothing)
i=0

res=0
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        img=cv2.flip(img, 0)
        img=cv2.flip(img, 1)
        scale_percent = 50
        
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        
        # dsize
        dsize = (width, height)
        
        # resize image
        img = cv2.resize(img, dsize)
    
        #blurred  = cv2.medianBlur(frame, 5)
        #blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        #
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
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
    
        # mask = cv2.GaussianBlur( mask, (3,3), 0)
        # mask = cv2.medianBlur(mask, 3)
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
            cv2.drawContours(img, contours, i, (0, 255, 0), 1)
        res = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("frame", img)
        cv2.imshow("mask", mask)
        cv2.imshow("res", res)
    
        if cv2.waitKey(1) & 0xFF == 27 :
            break
        grabResult.Release()
        
    # Releasing the resource    
camera.StopGrabbing()
cv2.destroyAllWindows()