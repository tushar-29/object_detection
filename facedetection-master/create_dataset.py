import cv2, sys, numpy, os
from time import sleep


MODEL = 'haarcascade_frontalface_default.xml'
D_FOLDER = 'datasets_file'
SUB_FOLDER = 'faces'

if not os.path.isdir(D_FOLDER):
    os.mkdir(D_FOLDER)

path = os.path.join(D_FOLDER, SUB_FOLDER)
if not os.path.isdir(path):
    os.mkdir(path)


(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(MODEL)
webcam = cv2.VideoCapture(0)

# The program loops until it has 30 images of the face.
count = 1
while count < 10:
    (_, img) = webcam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize)
    count += 1
    sleep(5)

    cv2.imshow('OpenCV', img)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
