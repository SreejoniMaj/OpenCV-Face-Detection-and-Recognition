import cv2
import os

cap=cv2.VideoCapture(0)

os.chdir('F:\Projects Career\Face Recognition Open CV\Training Images 2\Silpi')

count = 0
while True:
    ret,test_img=cap.read()
    if not ret :
        continue
    cv2.imwrite("frame%d.jpg" % count, test_img)     # save frame as JPG file
    count += 1
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection Tutorial ',resized_img)
    if cv2.waitKey(10) == 13:#wait until 'enter' key is pressed
        break


cap.release()
cv2.destroyAllWindows