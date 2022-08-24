# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 10:54:43 2022

@author: atakan
"""

import numpy as np
import cv2
rect = (0,0)
startPoint = False
endPoint = False
xt=0
yt=0
points=[]
x_old=0
y_old=0
def on_mouse(event,x,y,flags,params):

		global rect,startPoint,endPoint

		# get mouse click
		if event == cv2.EVENT_LBUTTONDOWN:
				rect = (x, y)

cap = cv2.VideoCapture(0)
	#waitTime = 50

	#Reading the first frame
(grabbed, frame) = cap.read()

while(cap.isOpened()):

		(grabbed, originalImage) = cap.read()
		frame=cv2.flip(originalImage, 0)
		frame=cv2.flip(frame, 1)
		cv2.namedWindow('frame')
		cv2.setMouseCallback('frame', on_mouse)    

		#drawing rectangle
		#if startPoint == True:
		if x_old != rect[0] and y_old!=rect[1]:
                    points.append((rect[0],rect[1]))
		cv2.circle(frame,(rect[0],rect[1]),3,(0,0,255),-1)        
		cv2.imshow('frame',frame)
		print(rect[0],rect[1])
		x_old=rect[0]
		y_old=rect[1]
		if cv2.waitKey(1) & 0xFF == 27:
			break

cap.release()
cv2.destroyAllWindows()


