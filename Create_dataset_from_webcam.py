import cv2
import sys

cpt = 0

vidStream = cv2.VideoCapture(0)
while True:
   
    ret, frame = vidStream.read()
    
    cv2.imshow("test window", frame)
	    
    cv2.imwrite(r"C:\Users\User\Desktop\Face-Recognition-master\train-images\2\image%04i.jpg" %cpt, frame)
    cpt+=1
    if cv2.waitKey(10)==ord('q'):
   		break
