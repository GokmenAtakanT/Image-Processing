# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:03:41 2020

@author: atakan
"""

import cv2
#img=cv2.imread("data/lena.jpg",0)
img=cv2.imread("prob.jpg",0)

print(img)
cv2.imshow('image',img)
k=cv2.waitKey(0)
if k==ord('s'):
    cv2.destroyAllWindows()
elif k== ord('b'):
    cv2.imwrite('prob.png',img)
    cv2.destroyAllWindows()
