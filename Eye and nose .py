import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0) #0 for web cam
if not cap.isOpened():
    raise IOError(" Cannot open wbcam")
while True:
    
    _ , frame = cap.read()
    c =cv.waitKey(1)
    if c ==27:
        break
    
    gray=cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    nose_casscade1=cv.CascadeClassifier("filepath")#nose casscade
    eye_classifier = cv.CascadeClassifier('filepath')#eye casscade
    faces=nose_casscade1.detectMultiScale(gray,1.3,10)
    eyes=eye_classifier.detectMultiScale(gray,1.3,5)
    
    for face in faces:
        x,y,w,h=face
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
    for (ex,ey,ew,eh) in eyes:
        centre=(int(ex+0.5*ew),int(ey+0.5*h))
        radius=int(0.3*(ew+eh))
        color=(0,255,0)
        thickness=3
        cv.circle(frame,centre,radius,(0,0,255),2) 
    
    
    
    cv.imshow("video",frame)
cap.release()
cv.destroyAllWindows()
