# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:51:39 2021

@author: gat06
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('data/cat_env2.jpg', cv2.IMREAD_GRAYSCALE)

# generating the kernels
vertical_emboss = np.array([[0,1,0],
                            [0,0,0],
                            [0,-1,0]])
southeast_northwest_emboss = np.array([[1,0,0],
                            [0,0,0],
                            [0,0,-1]])

horizontal_emboss = np.array([[0,0,0],
                            [1,0,-1],
                            [0,0,0]])

southwest_northeast_emboss = np.array([[0,0,1],
                            [0,0,0],
                            [-1,0,0]])

vertical_emboss_img = cv2.filter2D(img,-1,vertical_emboss)
southeast_northwest_emboss_img = cv2.filter2D(img,-1,southeast_northwest_emboss)
horizontal_emboss_img = cv2.filter2D(img,-1,horizontal_emboss)
southwest_northeast_emboss_img = cv2.filter2D(img,-1,southwest_northeast_emboss)

titles = ['Original','Vertical_emboss', 'Southeast_northwest_emboss', 'horizontal_emboss ', 'southwest_northeast_emboss_img']
images = [img, vertical_emboss_img, southeast_northwest_emboss, horizontal_emboss_img , southwest_northeast_emboss_img]

for i in range(2):
    plt.subplot(2, 1, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()