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
from imageio import get_writer

def find_nearest_with_cost(spirx:np.ndarray, spiry:np.ndarray, pixelx:int, pixely:int, current:int,prev_idx:int=None) -> int:
    dist2 = (spirx - pixelx)**2 + (spiry - pixely)**2
    if prev_idx==None :
        idx = np.argsort(dist2.flatten())[0]
    else:
        # Calculate cost
        past_dist2 = (spirx - int(spirx[prev_idx]))**2 + (spiry - int(spiry[prev_idx]))**2
        weighted_dist2 = dist2 + 5*past_dist2
        closest3 = np.argsort(weighted_dist2.flatten())[:3]
        costs = abs(closest3 - prev_idx)
        # Select idx from cost
        idx_temp = costs.argmin()
        idx = closest3[idx_temp]
    return idx

veriler = pd.read_excel('coclea_data_road_final3.xlsx')

data_ma=np.float64(veriler)
phi = data_ma[:,2:3]
X_ma = (data_ma[:,0:1])-100
Y_ma = (data_ma[:,1:2])
#teta=data[:,6:7]
phi_ma_var=[]
X_ma_var=[]
Y_ma_var=[]
previous = time()
for i in range(1,len(X_ma)):
    numx=np.linspace(int(X_ma[i-1]),int(X_ma[i]),100)
    numy=np.linspace(int(Y_ma[i-1]),int(Y_ma[i]),100)
    numphi=np.linspace(int(phi[i-1]),int(phi[i]),100)
    X_ma_var.append(numx)
    Y_ma_var.append(numy)
    phi_ma_var.append(numphi)


X_ma_var=np.reshape(np.float64(X_ma_var),(np.shape(X_ma_var)[0]*np.shape(X_ma_var)[1],1))
Y_ma_var=np.reshape(np.float64(Y_ma_var),(np.shape(Y_ma_var)[0]*np.shape(Y_ma_var)[1],1))
phi_ma_var=np.reshape(np.float64(phi_ma_var),(np.shape(phi_ma_var)[0]*np.shape(phi_ma_var)[1],1))


kf = cv2.KalmanFilter(4, 2)
kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1
kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
kf.measurementNoiseCov=np.array([[1, 1], [1,1]], np.float32) * 1
kf.errorCovPre= np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) * 1
kf.errorCovPost= np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], np.float32) *1
kf.statePost = 1 * np.random.randn(4, 2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
#fgbg = cv.bgsegm.BackgroundSubtractorGMG()
fgbg = cv2.createBackgroundSubtractorMOG2(history = 2000,varThreshold = 2000,detectShadows=True)
#fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)


center = (int(X_ma[0:1]),int(Y_ma[0:1]))
cur_old=0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_coclea_basler.avi', fourcc, 20.0, (640,  480))
valxp=[0 for i in range(55)]
valyp=[0 for i in range(55)]
teta_val=[0,]
error_tra=100
i=0
past=0
errorx_past=0
errory_past=0
prev_val=[]
prev = None
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
writer = get_writer(
       'output-filename.mkv',  # mkv players often support H.264
        codec='libx264',  # When used properly, this is basically "PNG for video" (i.e. lossless)
        quality=None,  # disables variable compression
        ffmpeg_params=[  # compatibility with older library versions
            '-preset',   # set to fast, faster, veryfast, superfast, ultrafast
            'fast',      # for higher speed but worse compression
            '-crf',      # quality; set to 0 for lossless, but keep in mind
            '24'         # that the camera probably adds static anyway
        ]
)

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    current = time()- previous

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
        dt_ch=current-past
        #mask = cv2.medianBlur(img, 21)
    
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #mask= clahe.apply(mask)
        if current <= 5:
            mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            l_b = np.array([0, 0, 0])
            u_b = np.array([255, 255, 100])
            mask = cv2.inRange(mask, l_b, u_b)
            mask = cv2.equalizeHist(mask)
            mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=6)
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)), iterations=9) 
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
        else:
            for k in range(6):
                mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                l_b = np.array([0, 0, 0])
                u_b = np.array([255, 255, 100])
                mask = cv2.inRange(mask, l_b, u_b)
                mask = cv2.equalizeHist(mask)
                mask = cv2.equalizeHist(mask)
        
                mask = cv2.medianBlur(mask, 15)
                mask = cv2.GaussianBlur(mask, (7+k*2, 7+k*2), 0)
                mask = fgbg.apply(mask)
                #mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1)), iterations=1)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT,(3+k,3+k)), iterations=7)         
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contours) < 2 :
                    break
        
        areas = [cv2.contourArea(c) for c in contours]
        if  len(areas)>0:
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            # Calculate the area of each contour
            area = cv2.contourArea(cnt)
            # Ignore contours that are too small or too large
            (x_m,y_m),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x_m),int(y_m))
            radius = int(radius)
            measured = np.array([[np.float32(int(x_m))], [np.float32(int(y_m))]])
            kf.correct(measured)
            (predicted) = kf.predict()
            x1=int(predicted[0])
            x2=int(predicted[1])
            x1_p=x1
            y1_p=x2
            valxp.append(x1_p)
            valyp.append(y1_p)
    
        x1_pa=np.mean(valxp[-20:])
        y1_pa=np.mean(valyp[-20:])
    
        xt=round((0.03849*float(x1_pa)-19.0525),1)
        yt=round((0.0388*float(y1_pa)-9.7),1)
        
        #value=(xt,yt)
    
        idx = find_nearest_with_cost(X_ma_var,Y_ma_var, x1_pa, y1_pa,current, prev_idx=prev)
        prev=idx
        orijin=(495,250)
        for j in range(8):
            cv2.line(img, (orijin[0],orijin[1]), (int(orijin[0]-orijin[0]*np.cos(j*np.pi/4)),int(orijin[1]+orijin[0]*np.sin(j*np.pi/4))), (0,255,0),2)
    
        for i in range(0,len(X_ma),1):
            cv2.circle(img,(int(X_ma[i]),int(Y_ma[i])),1,(0,0,255),2)
    
        cv2.circle(img,(int(X_ma_var[idx]),int(Y_ma_var[idx])),20,(0,255,0),2)
        cv2.circle(img,(orijin),1,(255,0,0),2)
        filename='phi_func.sav'
        model_direct=joblib.load(open(filename,'rb'))
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import mean_squared_error, r2_score

        # Preprocess inputs to be polynomial
        poly = PolynomialFeatures(degree=9)
        X_data=np.array(float(phi_ma_var[idx]))
        X_data=np.reshape(X_data,(-1, 1))
        poly_X = poly.fit_transform(X_data)
        pos=model_direct.predict(poly_X)
        
        value=(pos,float(phi_ma_var[idx]))
        cv2.putText(img, str(value), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('FG MASK Frame', mask)
        cv2.imshow('Frame', img)
        out.write(img)
        cur_old=current
        writer.append_data(img)
        if  cv2.waitKey(1) == 27:
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()