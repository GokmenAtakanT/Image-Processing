# -*- coding: utf-8 -*-
"""
Created on Tue May 26 22:18:42 2020

@author: gat06
"""


import matplotlib.pylab as plt
import cv2
import numpy as np

image = cv2.imread('data/road.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('image',image)
print(image.shape)
height = image.shape[0]
width = image.shape[1]

region_of_interest_vertices = [
    (0, height),
    (width/2, height/2),
    (width, height)
]

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    channel_count = img.shape[2]
    match_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cropped_image = region_of_interest(image,
                np.array([region_of_interest_vertices], np.int32),)

plt.imshow(cropped_image)

plt.show()
cv2.imshow('cropped_image',cropped_image)
k = cv2.waitKey(0)
cv2.destroyAllWindows()
