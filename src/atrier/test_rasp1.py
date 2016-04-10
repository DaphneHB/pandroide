import cv2
import numpy as np
import picamera

img = cv2.imread('tag2.jpg',0)
kernel = np.ones((5,5),np.uint8)

img2 = cv2.resize(img,(700,700))
ret,thresh = cv2.threshold(img2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#erosion = cv2.erode(thresh,kernel,iterations=1)

cv2.imshow('img',img2)
cv2.imshow('thresh',thresh)
#cv2.imshow('erodee',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
