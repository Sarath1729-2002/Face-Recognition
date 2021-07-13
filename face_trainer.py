import os
import cv2
import numpy as np
from PIL import Image as im
import pickle as p



x_train = []
y_labels = []
cur_id = 0
label_id = {}

base_dir = os.path.dirname(os.path.abspath(__file__))

img_dir = os.path.join(base_dir, "images")

recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')


for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path)
            if not label in label_id:
                label_id[label] = cur_id
                cur_id += 1

            id_ = label_id[label]

            #print(label_id)

            pil_image = im.open(path).convert("L")
            size = (550, 550)
            final_img = pil_image.resize(size, im.ANTIALIAS)
            arr = np.array(pil_image, "uint8")
            #print(arr)
            faces = face_cascade.detectMultiScale(arr, 1.5, 5)

            for (x, y, w, h) in faces:
                roi = arr[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

with open("labels.pickle", 'wb') as f:
    p.dump(label_id, f)

recognizer.train(x_train, np.array(y_labels))

recognizer.save("trainer.yml")


