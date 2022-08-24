# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:08:24 2020

@author: atakan
"""

import cv2
import numpy as np
template = cv2.imread("data/lagrange/lagrange_point.png", 0)
cv2.imshow("template", template)
w, h = template.shape[::-1]    
for i in range(1,2):
    img = cv2.imread('data/lagrange/'+str(i)+'.png')
    #cv2.imshow("original img", img)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(grey_img, template,cv2.TM_CCORR_NORMED )
    #cv2.TM_CCOEFF
    #cv2.TM_CCOEFF_NORMED
    #cv2.TM_CCORR
    #cv2.TM_CCORR_NORMED
    #cv2.TM_SQDIFF
    #cv2.TM_SQDIFF_NORMED
    print(res)
    threshold=0
    for i in range(np.shape(res)[0]):
        for j in range(np.shape(res)[1]):
            tmax=max(res[i:i+1,j:j+1])
            if threshold < tmax:
                threshold=tmax
            else:
                threshold=threshold
    loc = np.where(res >= threshold)
    print(loc)
    d1=int(sum(loc[0])/np.shape(loc)[1])
    d2=int(sum(loc[1])/np.shape(loc)[1])
    

    #for pt in zip(*loc[::-1]):
    cv2.rectangle(img, (d2,d1), (d2 + w, d1 + h), (0, 0, 255), 1)
    
    cv2.imshow("img", img)
    cv2.imwrite('lagrange templated img.png',img)
    cv2.destroyAllWindows()
    #cv2.imshow("lagrange_point", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()













