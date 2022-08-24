# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:07:48 2021

@author: gat06
"""

import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/smarties.png',cv2.COLOR_BGR2GRAY) # queryImage
detecter = cv2.xfeatures2d.SURF_create(500)
kp, des = detecter .detectAndCompute(img,None)
print (detecter.hessianThreshold)
detecter .hessianThreshold = 50000
kp, des = detecter.detectAndCompute(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()