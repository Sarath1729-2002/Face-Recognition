import cv2
from pathlib import Path

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)


print("Enter your name :")
fold = input()

count = 1

def saveimage(img, username, imgId):
    Path("images/{}".format(username)).mkdir(parents =True, exist_ok = True)
    cv2.imwrite("images/{}/{}.png".format(username, imgId), img)



while True:
    ret, frame = vc.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x ,y ,w ,h) in faces :

        img = cv2.rectangle(frame, (x, y), (x + w, y + h), [0,0,255], 2)
        pic = gray[y :y+h, x: x+w]

        cv2.imshow("Frame", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') :
        if count <= 100:
            saveimage(img, fold, count)
            count += 1
        else:
            break
    elif key == ord('q'):
        break
vc.release()
cv2.destroyAllWindows()




















