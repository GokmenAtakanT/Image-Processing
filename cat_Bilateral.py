# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:25:38 2021

@author: gat06
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/cat_env2.jpg', cv2.IMREAD_GRAYSCALE)
bilateral = cv2.bilateralFilter(img, 15, 75, 75)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(bilateral,cmap = 'gray')
plt.title('Bilateral Fİlter'), plt.xticks([]), plt.yticks([])

plt.show()