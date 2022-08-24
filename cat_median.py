# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:22:28 2021

@author: gat06
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imutils.video import VideoStream
import imutils

greenLower = (42, 50, 0)
greenUpper = (100, 255, 255)

while True:

    img = cv2.imread('data/cat_env_img6.jpg')
    
    blur = cv2.medianBlur(img, 5)
    blur = cv2.GaussianBlur(blur, (3, 3), 0)
    hsv = cv2.cvtColor(blur , cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=6) 
    res = cv2.bitwise_and(img, img, mask=mask)
    filename = 'savedImage.jpg'
    invert = cv2.bitwise_not(mask)
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

    #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    #text = "x: " + str(x) + ", y: " + str(y)

# Using cv2.imwrite() method
# Saving the image
    cv2.imwrite('data/cat/erodeFrame.jpg', img)
    cv2.imwrite('data/cat/erodeMask.jpg', mask)
    cv2.imwrite('data/cat/erodeInvert.jpg', invert)
    cv2.imwrite('data/cat/erodeResult.jpg', res)
    #☺cv2.circle(img,center,5,(0,0,255),2)
    cv2.imshow("frame", img)
    cv2.imshow("mask", mask)
    cv2.imshow("invert", invert)
    cv2.imshow("res", res)

    key = cv2.waitKey(1)
    if key == 27:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()