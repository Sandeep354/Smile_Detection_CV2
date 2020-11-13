# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:02:40 2020

@author: LENOVO
"""
# Face Recognition

# Import the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Defining the function that will do the detection
def detect(gray, frame):
    # gray - take image in black and white coz cascades work on grayscale images
    # frame - original image from camera
    
    # Get the cordinates of rectangles to detect the face (x,y,w-width,h-height)
    
    # (gray image, scale down factor, min # of accepted zones/neighbours around the pixel)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # This will return a tuple of (x,y,w,h)
    
    # Iterate through the multiple faces detected
    for (x, y, w, h) in faces:
        
        # (the frame, cord of upper right of rect, cord of lower right of rect, color in RGB, thickness of edges of the rectangle)
        face_rect = cv2.rectangle(frame, (x, y), (x+w, y+h), (173, 216, 230), 2)
        cv2.putText(face_rect, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        face_rect
        
        # Now detect eyes in reference to the face (inside this rectangle)
        # We will call it roi (region of interest)
        roi_gray = gray[y:y+h, x:x+w] #gray for eyes
        roi_color = frame[y:y+h, x:x+w] #colored for eyes (realtime)
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        # Iterate through eyes to draw th rectangle
        for (ex, ey, ew, eh) in eyes:
            eyes_rect = cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            # here the frame is now changed to that of the face - roi_color
            cv2.putText(eyes_rect, 'Eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            eyes_rect
            
    # return original frame with rectangles that detected the face and eyes
    return frame

# Doing some face recognition with the Webcam
# 0-internal camera of the laptop/pc, 1-external webcam
video_capture = cv2.VideoCapture(1)

# Apply detect function on all the frames captured by webcam
while True:
    # Get the last frame coming from the webcam
    # Read method return 2 element and we only want the last frame which is the second element (so we use _,variable = ..)
    _, frame = video_capture.read()
    
    # Now to transform the image from colored --> black and white (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray, frame)
    # Display all the processed images in an animated way with rectangles and all 
    cv2.imshow('Video', canvas)    
    
    # Stopping the webcam
    # Stop if we press 'q' on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Turn off the webcam
video_capture.release()
# Destroy the window where all the images are displayed
cv2.destroyAllWindows()
