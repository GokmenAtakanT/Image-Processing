# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:25:17 2020

@author: atakan
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/cat_env.jpg', cv2.IMREAD_GRAYSCALE)
_, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((5,5), np.uint8)

dilation = cv2.dilate(img, kernal, iterations=3)
erosion = cv2.erode(img, kernal, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernal)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernal)
mg = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernal)
th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernal)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
images = [img, mask, dilation, erosion, opening, closing, mg, th]

for i in range(4):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()