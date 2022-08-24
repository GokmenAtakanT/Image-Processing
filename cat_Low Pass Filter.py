# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:33:07 2021

@author: gat06
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/cat_env2.jpg', cv2.IMREAD_GRAYSCALE)


kernel = np.ones((3,3),np.float32)/9
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img, 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst, 'gray'),plt.title('Low Pass Filter ')
plt.xticks([]), plt.yticks([])
plt.show()