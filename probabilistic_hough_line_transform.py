# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:50:08 2020

@author: gat06
"""


import cv2
import numpy as np
img = cv2.imread('data/sudoku.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
cv2.imshow('edges', edges)
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
xval=[]
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imshow('image', img)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

xval=np.concatenate((lines[:,:,0:1], lines[:,:,2:3]), axis=1)