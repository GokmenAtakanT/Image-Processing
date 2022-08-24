# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:30:37 2021

@author: gat06
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/cat_env2.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img,150,255)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()