# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:49:29 2020

@author: atakan
"""

import cv2
import numpy as np
import datetime
cap = cv2.VideoCapture(0)

        
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(cap.isOpened())
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.flip(frame,1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text='WIDTH :' + str(cap.get(3)) + 'Height :' + str(cap.get(4))
        datet= str(datetime.datetime.now())
        frame=cv2.putText(frame,datet,(10,50),font,1,(0,255,255),2,cv2.LINE_AA)
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') :
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()









