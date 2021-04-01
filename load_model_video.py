import numpy as np
import cv2
import os
import matplotlib.pyplot as pt


import faceRecognition as fr

from datetime import date

today = date.today()

presents=[]
conf=[]
f=open("Attendance.txt","a+")
today_d="Date:- "+str(today);
f.write(today_d)
f.write("\n")
f.close()

def mark(name):
    f= open("Attendance.txt","a+")
    if name not in presents:
        presents.append(name)
        s=name+" - present \n"
        f.write(s)
        f.write("\n\n")
        f.close()

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\User\Desktop\Face-Recognition-master\trainingData.yml')

cap=cv2.VideoCapture(0)

name={0:"Indu Rawat",1:"Devina Negi",3:"Khushi Sharma"}


while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("face Detected: ",faces_detected)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print("Confidence :",confidence)
        conf.append(confidence)
        print("label :",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        mark(predicted_name)
        if(confidence>70):
            continue
        
        fr.put_text(test_img,predicted_name,x,y)

    resized_img=cv2.resize(test_img,(1000,700))

    cv2.imshow("face detection ", resized_img)
    if cv2.waitKey(10)==ord('q'):
        break
pt.style.use('ggplot')
pt.plot(conf,'-o',label='Confidance variation with time')
pt.legend()
pt.show()