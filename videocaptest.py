import cv2

face_cascade = cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0) 

while True:
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,8)
    
    for(x,y,w,h) in faces:
        print (faces)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,150,200),3)
         
    cv2.imshow('real time video capture',img)
    k=cv2.waitKey(10) 
    if k == 13:#wait until 'enter' key is pressed
        break
    
cap.release()
cv2.destroyAllWindows()    