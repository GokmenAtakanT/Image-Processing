# -*- coding: utf-8 -*-
"""
Created on Fri May 22 19:49:40 2020

@author: atakan
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("data/lena.jpg", cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
edges = cv2.Canny(img,100,200)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)
lap_cannycom = cv2.bitwise_or(lap, edges)


titles = ['image', 'Laplacian', 'sobelX', 'sobelY', 'sobelCombined', 'Canny','lap_cannycom']
images = [img, lap, sobelX, sobelY, sobelCombined, edges,lap_cannycom]
for i in range(7):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()












