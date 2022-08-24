# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:22:23 2021

@author: gat06
"""


from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
 
# Project: Object Tracking
# Author: Addison Sears-Collins 
# Website: https://automaticaddison.com
# Date created: 06/13/2020
# Python version: 3.7
 
def main():
    """
    Main method of the program.
    """
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
 
    # Create a VideoCapture object
 
    # Create the background subtractor object
    # Use the last 700 video frames to build the background
    back_sub = cv2.createBackgroundSubtractorMOG2(history=1000, 
        varThreshold=30, detectShadows=True)
 
    # Create kernel for morphological operation
    # You can tweak the dimensions of the kernel
    # e.g. instead of 20,20 you can try 30,30.
    kernel = np.ones((30,30),np.uint8)
 
    while(True):
 
        # Capture frame-by-frame
        # This method returns True/False as well
        # as the video frame.
        fg_mask1 = back_sub.apply(frame1)
        fg_mask2 = back_sub.apply(frame2)

        diff = cv2.absdiff(fg_mask1, fg_mask2)
        #diff = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
 
    
 
        # Use every frame to calculate the foreground mask and update
        # the background
        #fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)

        # Close dark gaps in foreground object using closing
        fg_mask = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
        
        fg_mask = cv2.GaussianBlur(fg_mask, (9,9), 0)
        # Remove salt and pepper noise with a median filter
        fg_mask = cv2.medianBlur(fg_mask, 9) 
         
        # Threshold the image to make it either black or white
        _, fg_mask = cv2.threshold(fg_mask,120,255,cv2.THRESH_BINARY)
        fg_mask = cv2.dilate(fg_mask, None, iterations=3)

        # Find the index of the largest contour and draw bounding box
        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(fg_mask_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
 
        # If there are no countours
        if len(areas) < 1:
 
            # Display the resulting frame
            cv2.imshow('frame',frame2)
 
            # If "q" is pressed on the keyboard, 
            # exit this loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
            # Go to the top of the while loop
            continue
 
        else:
            # Find the largest moving object in the image
            max_index = np.argmax(areas)
        frame1 = frame2
        ret, frame2 = cap.read()
        # Draw the bounding box
        cnt = contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
 
        # Draw circle in the center of the bounding box
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame2,(x2,y2),4,(0,255,0),-1)
 
        # Print the centroid coordinates (we'll use the center of the
        # bounding box) on the image
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame2, text, (x2 , y2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         
        # Display the resulting frame
        cv2.imshow('frame',frame2)
 
        # If "q" is pressed on the keyboard, 
        # exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(x2,y2)
 
    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    print(__doc__)
    main()