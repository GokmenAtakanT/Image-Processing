# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:36:59 2021

@author: gat06
"""

import tkinter
import cv2
from PIL import Image,ImageTk


root = tkinter.Tk()

root.geometry("1000x500+200+0")
w = 1000
h = 630

capture = tkinter.Canvas(root, bd=2, bg="blue", height=h, width=w)
capture.grid(column = 0, row = 0)
video = None
frame = None
img=None
show=None
begin = False
def start_capture(event):
    global begin,frame,img,root,show,capture,video    
    video = cv2.VideoCapture(0)
    begin = True
    while begin:
        check, frame = video.read()

        img = Image.fromarray(frame)
        w,h = img.size
        img = img.resize((w*2,h*2))
        show = ImageTk.PhotoImage(img)
        capture.create_image(0,0,anchor=tkinter.NW,image=show)
        root.update()

def stop_capture(event):
    global video,begin
    begin = False
    video.release()

start = tkinter.Button(root, text='Start')
start.grid(column = 0, row = 2)
start.bind('<Button-1>', start_capture)  
stop = tkinter.Button(root, text='Stop')
stop.grid(column = 1, row = 2)
stop.bind('<Button-1>', stop_capture)  

root.mainloop()