# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:02:50 2020

@author: gat06
"""
import numpy as np
import cv2 as cv
import matplotlib.pylab as plt

img = cv.imread('data/smarties.png')
output = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                          param1=50, param2=30, minRadius=0, maxRadius=0)
detected_circles = np.uint16(np.around(circles))
r_point=[]
x_point=[]
y_point=[]

for (x, y ,r) in detected_circles[0, :]:
    cv.circle(output, (x, y), r, (0, 0, 0), 3)
    cv.circle(output, (x, y), 2, (0, 255, 255), 3)
    r_point.append(r)
    x_point.append(x)
    y_point.append(y)


cv.imshow('image',img)
cv.imshow('output',output)
plt.imshow(output)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()