# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:11:31 2021

@author: gat06
"""

import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0);

cv2.namedWindow("Tracking")
cv2.createTrackbar("H", "Tracking", 0, 255, nothing)
cv2.createTrackbar("L", "Tracking", 0, 255, nothing)


while True:
    _, frame = cap.read()
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = cv2.getTrackbarPos("H", "Tracking")
    l = cv2.getTrackbarPos("L", "Tracking")
    ret, thresh = cv2.threshold(imgray, h, l, 0)

    cv2.imshow("frame", imgray)
    cv2.imshow("mask", thresh)

    key = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()