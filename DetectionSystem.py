import cv2
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
from tkinter import *
import tkinter as tk

root=tk.Tk()
root.minsize(width=300,height=300)
root.maxsize(width=500,height=500)

#face .xml tarin file
face_cascade=cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#function for face 
def detct_face(img):
    face_img=img.copy()
    face_react=face_cascade.detectMultiScale(face_img,scaleFactor=1.4,minNeighbors=5)
    for (x,y,w,h) in face_react:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(0,128,255),8)
    return face_img

#function for face videostream
def capture_face():
    cap=cv2.VideoCapture(0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # for windows user VideoWriter_fourcc = *'DIVX'
    #for MACOS or LINUX VideoWriter_Fourcc=*'XVID'
    writer=cv2.VideoWriter('video.Mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))#store mp4 file using writer function
    while True:
        ret,frame=cap.read()
        writer.write(frame)
        frame=detct_face(frame)
   
        cv2.imshow('video Face Detect',frame)
        if cv2.waitKey(1) & 0xFF==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

#train  eye .xml  file
eye_cascade=cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

#train fullbody .xml file
fullbody_cascade=cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')

#function for fullbody
def detect_fullbody(img):
    fullbody_img=img.copy()
    fullbody_react=fullbody_cascade.detectMultiScale(fullbody_img,scaleFactor=1.4,minNeighbors=5)
    for (x,y,w,h) in fullbody_react:
        cv2.rectangle(fullbody_img,(x,y),(x+w,y+h),(0,128,255),8)
    return fullbody_img


#function for fullbody videoCapture
def capture_fullbody():
    cap=cv2.VideoCapture(0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter('video.Mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))
    while True:
        ret,frame=cap.read()
        writer.write(frame)
        frame=detect_fullbody(frame)
   
        cv2.imshow('video Face Detect',frame)
        if cv2.waitKey(1) & 0xFF==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

#eye detection function
def detect_eye(img):
    eye_img=img.copy()
    eye_react=eye_cascade.detectMultiScale(eye_img,scaleFactor=1.2,minNeighbors=3)
    for (x,y,w,h) in eye_react:
        cv2.rectangle(eye_img,(x,y),(x+w,y+h),(0,128,255),8)
    return eye_img

#eye video detection function
def capture_eye():
    cap=cv2.VideoCapture(0)
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter('video.Mp4',cv2.VideoWriter_fourcc(*'DIVX'), 10, (width,height))
    while True:
        ret,frame=cap.read()
        writer.write(frame)
        frame=detect_eye(frame)
   
        cv2.imshow('video Face Detect',frame)
        if cv2.waitKey(1) & 0xFF==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()


root.title('Detection System By (Tushar Gharge)')
#root.iconbitmap("detectionfinal.png")

#detecton.png image is user for icon of project 
#set path according to store image
root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file='detection.png'))

# pick a .gif image file you have in the working directory
fname = "bg1.gif"
bg_image = tk.PhotoImage(file=fname)
# get the width and height of the image
w = bg_image.width()
h = bg_image.height()

# size the window so the image will fill it
root.geometry("%dx%d+50+30" % (w, h))

cv = tk.Canvas(width=w, height=h)
cv.pack(side='top', fill='both', expand='yes')

cv.create_image(0, 0, image=bg_image, anchor='nw')
#Label use of Detection system Aim
label1=Label(cv,text="                      Detection System                    ",bg="Blue",fg="Red",font=("Times",20),bd=3).pack()
Label(cv,text=" ").pack()
#Button call he capture_face function  for face Detection 
Button1=Button(cv,text="Face Detection",command=capture_face,font=("Times",16),height=1,width=18)
Button1.pack()
Label(cv,text=" ").pack()

#Button call he capture_face function  for face Detection 
Button2=Button(cv,text="Eye Detection",command=capture_eye, font=("Times",16),height=1,width=18)
Button2.pack()
Label(cv,text=" ").pack()

#Button call he capture_face function  for face Detection 
Button3=Button(cv,text="FullBody Detection",command=capture_fullbody,font=("Times",16),height=1,width=18)
Button3.pack()
Label(root,text=" ").pack()

root.mainloop()





