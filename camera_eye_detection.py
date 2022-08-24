# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:18:51 2020

@author: gat06
"""


import cv2
cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('img/myfacedetector3.xml')

#face_cascade = cv2.CascadeClassifier('img/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('img/haarcascade_eye_tree_eyeglasses.xml')
#cap = cv2.VideoCapture('output.avi')

while cap.isOpened():
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y , w ,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0 , 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'face',(x,y),font,1,(0,0,255),1,cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey ,ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(roi_color,'eye',(ex,ey),font,1,(0,0,255),1,cv2.LINE_AA)

    # Display the output
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()