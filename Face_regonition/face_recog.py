import cv2
import os
if not os.path.exists('find_me'):
    os.makedirs('find_me')

id = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('./yml/train.yml')
face_cascade_Path = "./yml/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(face_cascade_Path)
font = cv2.FONT_HERSHEY_SIMPLEX

names = {3: 'jimmy fermin', 6: 'Michael Dam', 51: 'shouvik',8: 'dashmat'}  # add a name into this list

predictName = 'jimmy-fermin'


# while True:
img = cv2.imread('./find_me/%s' %predictName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    )

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
    if (confidence < 100):
        # print(id)
        id = names[id]
        confidence = "  {0}%".format(round(100 - confidence))
    else:
        # Unknown Face
        id = "Not able to Match ?"
        confidence = "  {0}%".format(round(100 - confidence))

result = id
result2 = confidence

