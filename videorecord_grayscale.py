# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:50:52 2020

@author: atakan
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

        
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.set(3,1024)
cap.set(4,1024)

print(cap.isOpened())
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()