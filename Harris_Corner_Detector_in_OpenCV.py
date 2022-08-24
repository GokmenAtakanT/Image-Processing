# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:13:41 2020

@author: gat06
"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('data/chessboard.png')

cv.imshow('img', img)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)

dst = cv.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]
cv.imshow('dst', img)
plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.show()
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()