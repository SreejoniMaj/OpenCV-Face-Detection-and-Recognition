import cv2
import os
import numpy as np
import faceRecognition as fr

test_img = cv2.imread("Test Images 2\kareena3.png")
faces, grey_img = fr.faceDetection(test_img)
print("faces_detected: ", faces)
if len(faces) < 1:
    print("No faces detected")
    exit()

# resized_img=test_img

# below lines are for trainig the model with the training images just once

# faces_detected,faceID=fr.labels('Training Images 2')
# face_recognizer=fr.train_classifier(faces_detected,faceID)
# face_recognizer.write('TrainingData3.yml')

# below lines are run after the training data is ready and we don't need to retrain the model anymore

face_recognizer = cv2.face.LBPHFaceRecognizer.create()
face_recognizer.read("TrainingData3.yml")


name = {0: "Shah Rukh Khan", 1: "Kareena Kapoor", 2: "Silpi", 3: "Sreejoni"}

for face in faces:
    (x, y, w, h) = face
    roi_grey = grey_img[y : y + w, x : x + h]
    label, confidence = face_recognizer.predict(roi_grey)
    print("Confidence: ", confidence)
    print("Predicted Label : ", name[label])

    predicted_label = name[label]
    if label == 0 or label == 1:
        fr.bounding_box(test_img, face)
        # if(confidence>37):
        #     continue
        fr.put_label(test_img, predicted_label, x - 15, y)
        resized_img = cv2.resize(test_img, (600, 500))
        
    else:
        fr.bounding_box2(test_img, face)
        if confidence > 60:
            continue
        fr.put_label2(test_img, predicted_label, x, y)
        resized_img = cv2.resize(test_img, (600, 600))
        
    
cv2.imshow("Face Recognition", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows


