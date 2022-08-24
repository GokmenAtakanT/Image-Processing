# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:28:38 2021

@author: gat06
"""

import cv2
import numpy as np
cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create(nfeatures=150000)
    
    #keypoints_sift, descriptors = sift.detectAndCompute(img, None)
    #keypoints_surf, descriptors = surf.detectAndCompute(img, None)
    keypoints_orb, descriptors = orb.detectAndCompute(gray, None)
    
    gray = cv2.drawKeypoints(gray, keypoints_orb, None)
    cv2.imshow('image',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
