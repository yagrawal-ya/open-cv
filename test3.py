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
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cap = cv2.VideoCapture("lane_vgt.mp4")

while(1):

    ret, frame = cap.read()
    gray = convert_hls(frame)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    final = cv2.Canny(blur,90,140,apertureSize = 3)
    vertices = select_region(final)
    final = filter_region(final,vertices)
    
    kernel = np.ones((5,5), np.uint8)
 
# The first parameter is the original image,
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image. 
    final = cv2.erode(final, kernel, iterations=10)


    #sobelx = cv2.Sobel(edges,cv2.CV_64F,1,0,ksize=5)
    #sobelx64f = cv2.Sobel(edges,cv2.CV_64F,1,0,ksize=5)
    #abs_sobel64f = np.absolute(sobelx64f)
    cv2.imshow('res',final)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

