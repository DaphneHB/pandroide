# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:30:18 2016

@author: emilie
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 01:04:35 2016

@author: emilie
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg')

#tuple of number of rows, columns and channels

print img.shape  

#Total number of pixels i
print img.size

#img.dtype is very important while debugging
# because a large number of errors in OpenCV-Python code is caused by invalid datatype.
print img.dtype
#copie zone carree de l image dans image
ball = img[200:256, 200:256]
img[0:56, 0:56] = ball
cv2.imwrite('testcopie.png',img)

# pour le niveau de Bleu
print img[1][2][0]
# pour le niveau de Vert
print img[1][2][1]
# pour le niveau de Rouge
print img[1][2][2]


img = cv2.imread('lena.jpg',0)
img2 = cv2.imread('tag.png',0)

edges = cv2.Canny(img,100,200)
cv2.imwrite('lenaedges.png',edges)

edges2 = cv2.Canny(img2,100,200)
cv2.imwrite('tagedges.png',edges2)

edges = cv2.Canny(img,50,300)
cv2.imwrite('lenaedges2.png',edges)

edges2 = cv2.Canny(img2,50,300)
cv2.imwrite('tagedges2.png',edges2)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
cv2.imwrite('lenalaplacian.png',laplacian)

laplacian2 = cv2.Laplacian(img2,cv2.CV_64F)
cv2.imwrite('taglaplacian.png',laplacian2)

dst2 = cv2.convertScaleAbs(laplacian2)
cv2.imwrite('tagdst.png',dst2)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
cv2.imwrite('lenasobelx3.png',sobelx)

sobelx2 = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=3)
cv2.imwrite('tagsobelx3.png',sobelx2)

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
cv2.imwrite('lenasobely3.png',sobely)

sobely2 = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=3)
cv2.imwrite('tagsobely3.png',sobely2)

sobely = cv2.Sobel(sobelx,cv2.CV_64F,0,1,ksize=3)
cv2.imwrite('lenasobelxsobely3.png',sobely)

sobely2 = cv2.Sobel(sobelx2,cv2.CV_64F,0,1,ksize=3)
cv2.imwrite('tagsobelxsobely3.png',sobely2)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
cv2.imwrite('tagsobelx5.png',sobelx)

sobelx2 = cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=5)
cv2.imwrite('tagsobelx5.png',sobelx2)

sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
cv2.imwrite('lenasobely5.png',sobely)

sobely2 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
cv2.imwrite('tagsobely5.png',sobely2)

sobely = cv2.Sobel(sobelx,cv2.CV_64F,0,1,ksize=5)
cv2.imwrite('lenasobelxsobely5.png',sobely)

sobely2 = cv2.Sobel(sobelx2,cv2.CV_64F,0,1,ksize=5)
cv2.imwrite('tagsobelxsobely5.png',sobely2)

#Scharr 

#sqrBoxFilter

# spatialGradient

#cv::spatialGradient 

#sift


#imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret,thresh = cv2.threshold(imgray,127,255,0)
#image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print contours
