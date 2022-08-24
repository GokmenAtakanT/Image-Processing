# -*- coding: utf-8 -*-
"""
Created on Sat May 23 02:17:05 2020

@author: atakan
"""

import cv2
import numpy as np
img = cv2.imread("data/lena.jpg")
layer = img.copy()
gaussian_pyramid_list = [layer]
#lr1 = cv2.pyrDown(img)
#hr1 = cv2.pyrUp(img)
#cv2.imshow("original image ",img)
#cv2.imshow("original lower image ",lr1)
#cv2.imshow("original upper image ",hr1)

for i in range(6):
    layer = cv2.pyrDown(layer)
    gaussian_pyramid_list.append(layer)
    #cv2.imshow(str(i), layer)

layer = gaussian_pyramid_list[5]
cv2.imshow('upper level Gaussian Pyramid', layer)
laplacian_pyramid_list = [layer]

for i in range(5, 0, -1):
    gaussian_extended = cv2.pyrUp(gaussian_pyramid_list[i])
    laplacian = cv2.subtract(gaussian_pyramid_list[i-1], gaussian_extended)
    cv2.imshow(str(i), laplacian)

cv2.imshow("Original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()