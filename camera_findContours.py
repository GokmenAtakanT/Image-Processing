# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:02:31 2021

@author: gat06
"""



import numpy as np
import cv2


cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, img = cap.read()
    #img = cv2.imread('data/baseball.png')
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 20, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #print("Number of contours = " + str(len(contours)))
    #print(contours[0])
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    #cv2.drawContours(imgray, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Image', img)
    #cv2.imshow('Image GRAY', imgray)
    #cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()