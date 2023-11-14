import cv2
import os
import numpy as np


def faceDetection(test_img):
    grey_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')
    faces=face_cascade.detectMultiScale(grey_img,scaleFactor=1.32,minNeighbors=7)
    
    return faces,grey_img


def labels(directory):
    faces=[]
    faceID=[]
    
    for path,subdir,filenames in os.walk(directory):
        for filename in filenames:
            
            name=os.path.basename(path)
            img_path=os.path.join(path,filename)
            print("Image path : ",img_path)
            print("Name ",name)
            
            id={"Shah Rukh Khan":0, "Kareena Kapoor":1, "Silpi":2, "Sreejoni":3}
            
            
            print("ID : ",id[name])
            test_img=cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            
            faces_detected,grey_img=faceDetection(test_img)
            if len(faces_detected)!=1:
                continue
            (x,y,w,h)=faces_detected[0]
            roi_grey=grey_img[y:y+w,x:x+h]
            faces.append(roi_grey)
            faceID.append(id[name])
            
    return faces,faceID
            
    
    

def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer.create()
    face_recognizer.train(faces,np.array(faceID))
    
    return face_recognizer


def bounding_box(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,150,200),thickness=2)
    
    
def bounding_box2(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,150,200),thickness=15)
    
def put_label(test_img,label,x,y):
    cv2.putText(test_img,label,(x,y),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),2)
    
def put_label2(test_img,label,x,y):
    cv2.putText(test_img,label,(x,y),cv2.FONT_HERSHEY_TRIPLEX,8,(0,0,255),7)