# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:39:32 2021

@author: gat06
"""

import cv2
def show_webcam():
    scale=20

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, image = cam.read()
        image = cv2.flip(image, 1)
        #get the webcam size
        height, width, channels = image.shape

        #prepare the crop
        centerX,centerY=int(height/2),int(width/2)
        radiusX,radiusY= int(scale*height/100),int(scale*width/100)

        minX,maxX=centerX-radiusX,centerX+radiusX
        minY,maxY=centerY-radiusY,centerY+radiusY

        cropped = image[minX:maxX, minY:maxY]
        resized_cropped = cv2.resize(cropped, (width, height)) 

        cv2.imshow('my webcam', resized_cropped)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break  # esc to quit

        if cv2.waitKey(1) & 0xFF == ord('w'): 
            scale =scale + 5  # +5

        if cv2.waitKey(1) & 0xFF == ord('s'): 
            scale =scale - 5  # +5

    cv2.destroyAllWindows()


def main():
    show_webcam()


if __name__ == '__main__':
    main()