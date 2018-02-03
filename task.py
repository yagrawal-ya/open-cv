import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("lane_vgt.mp4")

while(1):

    ret, frame = cap.read()
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    mask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_not(mask)

    
    res = cv2.bitwise_and(frame,frame, mask= mask)
    #ret,thresh2res = cv2.threshold(res,0,255,cv2.THRESH_BINARY)
    thresh2res=cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(thresh2res,500,550)

        
    #cv2.imshow('frame',frame)
    cv2.imshow('res',dst)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()