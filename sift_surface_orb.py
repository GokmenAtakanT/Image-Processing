# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:03:53 2021

@author: gat06
"""

import cv2
import numpy as np

img = cv2.imread('data/smarties.png',cv2.COLOR_BGR2GRAY) # queryImage

#sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(nfeatures=150000)

#keypoints_sift, descriptors = sift.detectAndCompute(img, None)
#keypoints_surf, descriptors = surf.detectAndCompute(img, None)
keypoints_orb, descriptors = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints_orb, None)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()