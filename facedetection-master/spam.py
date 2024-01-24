import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)


while True:
    _, img = webcam.read()
    change_img = img
    # print(img)
    for i, rgb_list in enumerate(change_img):
        for j, rgb_value in enumerate(rgb_list):
            rgb_value = [rgb_value[0]*0.2986, rgb_value[1]*0.5870 , rgb_value[2]*0.0722]

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(change_img, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(change_img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', change_img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


webcam.release()
cv2.destroyAllWindows()
