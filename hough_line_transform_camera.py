# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:46:30 2021

@author: gat06
"""



import cv2
import numpy as np
cap = cv2.VideoCapture(1)

while cap.isOpened():
    _, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    #detected_circles = np.uint16(np.around(lines))

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


















