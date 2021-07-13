import numpy as np
import cv2
import pickle

face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"persons_name": 1}

with open( "labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+y]

        id_, conf = recognizer.predict(roi_gray)

        img_item = "my_image.png"
        cv2.imwrite(img_item, roi_gray)

        if conf >= 50:
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name ,(x,y), font, 1, color, stroke, cv2.LINE_AA)

        color = (255, 0 ,0)
        stroke = 2
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

