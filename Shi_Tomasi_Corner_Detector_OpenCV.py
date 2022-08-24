# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:43:55 2020

@author: gat06
"""


import numpy as np
import cv2 as cv

img = cv.imread('data/pic1.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 100, 0.001, 1)

corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 3, [255, 255, 0], -1)

cv.imshow('Shi-Tomasi Corner Detector', img)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()