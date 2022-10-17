import cv2
import os
from fastapi import UploadFile
import random

# Check if folder exists
if not os.path.exists('images'):
    os.makedirs('images')



face_cascade = cv2.CascadeClassifier('./yml/haarcascade_frontalface_default.xml')
count = 0
face_id = 69

personFileName =" Michael-Dam.jpg"

while(True):
    img = cv2.imread('./find_me/%s' %personFileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        count += 1
    # Save the captured image into the images directory

    # # Press Escape to end the program.
    k = cv2.waitKey(100) & 0xff
    if k < 30:
        break
    elif count >= 10:
        break





