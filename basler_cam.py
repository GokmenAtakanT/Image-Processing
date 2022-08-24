# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:34:47 2022

@author: atakan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:05:24 2022

@author: atakan
"""

from pypylon import pylon
import cv2
from matplotlib import pyplot as plt
import numpy as np
from time import time
import joblib
import pandas as pd



camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        img=cv2.flip(img, 0)
        img=cv2.flip(img, 1)
        scale_percent = 50
        
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        
        # dsize
        dsize = (width, height)
        
        # resize image
        img = cv2.resize(img, dsize)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

        cv2.imshow('Frame', img)
        if  cv2.waitKey(1) == 27:
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()