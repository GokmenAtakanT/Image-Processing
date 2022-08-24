# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:33:45 2020

@author: atakan
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

print(cap.isOpened())
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)
        
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out.write(frame)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()








