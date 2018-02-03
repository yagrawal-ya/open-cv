import cv2
import numpy as np

def threshold(img):
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_)
    return th2

cap=cv2.VideoCapture("lane_vgt.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #thresh = threshold(gray)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.imshow('Output',th2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



