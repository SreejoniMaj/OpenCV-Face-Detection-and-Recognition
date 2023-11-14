import cv2
import faceRecognition as fr

# face_cascade = cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')


face_recognizer = cv2.face.LBPHFaceRecognizer.create()
face_recognizer.read("TrainingData3.yml")

name = {0: "Shah Rukh Khan", 1: "Kareena Kapoor", 2: "Silpi", 3: "Sreejoni"}

cap=cv2.VideoCapture(0) 

while True:
    ret,img=cap.read()
    img_copy=img.copy()
    faces, grey_img = fr.faceDetection(img)
    print("faces_detected: ", faces)
    # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # faces=face_cascade.detectMultiScale(gray,1.3,8)
    
    # for(x,y,w,h) in faces:
    #     # print (faces)
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,150,200),3)
    
    for face in faces:
        (x, y, w, h) = face
        roi_grey = grey_img[y : y + h, x : x + w]
        label, confidence = face_recognizer.predict(roi_grey)
        print("Confidence: ", confidence)
        print("Predicted Label : ", name[label])

        predicted_label = name[label]
    
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,150,200),3)
        if confidence>85:
            continue
        cv2.putText(img,predicted_label,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)
            

    cv2.imshow('real time video capture',img)
    k=cv2.waitKey(10) 
    if k == 13:#wait until 'enter' key is pressed
        break
    
cap.release()
cv2.destroyAllWindows()  