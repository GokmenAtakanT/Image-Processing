# -*- coding: utf-8 -*-
"""
Created on Sun May 24 01:57:27 2020

@author: atakan
"""

import cv2

cap = cv2.VideoCapture('coclea_close2.mp4')

while cap.isOpened():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    _, thresh = cv2.threshold(blur, 2, 100, cv2.THRESH_BINARY)
    mask = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=3)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=5) 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area > 5e4 and 6e4 > area:
            continue

        # Draw each contour only for visualisation purposes
        cv2.drawContours(frame1, contours, i, (0, 255, 0), 1)
    cv2.imshow('Image', frame1)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # if ret==False:
    #     print("Can't receive frame (stream end?). Exiting ...")
    # break
cap.release()
cv2.destroyAllWindows()






