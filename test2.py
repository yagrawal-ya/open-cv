import cv2
import numpy as np

def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0, rows*0.95]
    top_left     = [cols*0, rows*0.40]
    bottom_right = [cols*1, rows*0.95]
    top_right    = [cols*1, rows*0.40] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return vertices

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

cap = cv2.VideoCapture("lane_vgt.mp4")

while(1):

    ret, frame = cap.read()
    frame=convert_hls(frame)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    final = cv2.Canny(blur,90,140,apertureSize = 3)
    lines = cv2.HoughLines(final,1,np.pi/180, 40)
    for r,theta in lines[0]:
         
    # Stores the value of cos(theta) in a
        a = np.cos(theta)
 
    # Stores the value of sin(theta) in b
        b = np.sin(theta)
     
    # x0 stores the value rcos(theta)
        x0 = a*r
     
    # y0 stores the value rsin(theta)
        y0 = b*r
     
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000*(-b))
     
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000*(a))
 
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000*(-b))
     
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000*(a))
     
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be 
    #drawn. In this case, it is red. 
        cv2.line(final,(x1,y1), (x2,y2), (255,255,255),2)
    
    
    vertices = select_region(final)
    final = filter_region(final,vertices)
    
    kernel = np.ones((1,1), np.uint8)
 
# The first parameter is the original image,
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image. 
    

    #sobelx = cv2.Sobel(edges,cv2.CV_64F,1,0,ksize=5)
    #sobelx64f = cv2.Sobel(edges,cv2.CV_64F,1,0,ksize=5)
    #abs_sobel64f = np.absolute(sobelx64f)
    cv2.imshow('res',final)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

