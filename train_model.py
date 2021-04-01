import numpy as np
import cv2
import os

import faceRecognition as fr


test_img=cv2.imread(r'C:\Users\User\Desktop\Face-Recognition-master\0.jpg')
test_img=cv2.imread(r'C:\Users\User\Desktop\Face-Recognition-master\1.jpeg')
test_img=cv2.imread(r'C:\Users\User\Desktop\Face-Recognition-master\3.jpeg')

faces_detected,gray_img=fr.faceDetection(test_img)
print("face Detected: ",faces_detected)


faces,faceID=fr.labels_for_training_data(r'C:\Users\User\Desktop\Face-Recognition-master\train-images')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:\Users\User\Desktop\Face-Recognition-master\trainingData.yml')

name={0:"Indu Rawat",1:"Devina Negi",3:"Khushi Sharma"}


for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("Confidence :",confidence)
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>70):
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows







