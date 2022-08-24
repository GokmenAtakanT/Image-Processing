# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:18:17 2020

@author: atakan
"""
import cv2
import numpy as np
img=cv2.imread("data/messi5.jpg",1)
img2=cv2.imread("data/opencv-logo.png",1)

b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))

ball= img[280:340,330:390]
img[273:333,100:160]=ball
img = cv2.resize(img,(512,512))
img2 = cv2.resize(img2,(512,512))

#dst = cv2.add(img,img2)
dst=cv2.addWeighted(img,0.9,img2,0.1,0)
cv2.imshow('image',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()