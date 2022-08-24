# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:24:18 2020

@author: atakan
"""

import cv2 as cv
import numpy as np

img = cv.imread('data/sudoku.png',0)
_, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2);
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);
while True:
    cv.imshow("Image", img)
    cv.imshow("THRESH_BINARY", th1)
    cv.imshow("ADAPTIVE_THRESH_MEAN_C", th2)
    cv.imshow("ADAPTIVE_THRESH_GAUSSIAN_C", th3)

    cv.waitKey(1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()