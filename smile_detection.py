# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:56:50 2020

@author: LENOVO
"""

import cv2

faces_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    
    faces = faces_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        
        face_rect = cv2.rectangle(frame, (x, y), (x+w, y+h), (173, 216, 230), 2)
        cv2.putText(face_rect, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        face_rect
        
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w] 
        
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.3, 22)
        
        for (ex, ey, ew, eh) in eyes:
            eyes_rect = cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            cv2.putText(eyes_rect, 'Eye', (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            eyes_rect
        
        for (sx, sy, sw, sh) in smile:
            smile_rect = cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            cv2.putText(smile_rect, 'Smily', (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            smile_rect
            
    return frame

video_capture = cv2.VideoCapture(1)

while True:
    _, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)    
    
    # Stopping the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Turn off the webcam
video_capture.release()
# Destroy the window where all the images are displayed
cv2.destroyAllWindows()
