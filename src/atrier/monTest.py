# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:32:34 2016

@author: 3200234
"""

import cv2
import numpy as np
import time

SEUIL = 150

img = cv2.imread("lena.jpg",0)


tdebut = time.time()
ret2,img2 = cv2.threshold(img,SEUIL,255,cv2.THRESH_BINARY)
tfin = time.time()
print 'Temps en CV2.THRESHOLD = {}s'.format(tfin-tdebut)

tdebut = time.time()
ret3,img3 = cv2.threshold(img,SEUIL,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
tfin = time.time()
print 'Temps en CV2.THRESHOLD+OTSU = {}s'.format(tfin-tdebut)

tdebut = time.time()
blur = cv2.GaussianBlur(img,(5,5),0)
ret4,img4 = cv2.threshold(blur,SEUIL,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
tfin = time.time()
print 'Temps en CV2.THRESHOLD+GAUSSIAN+OTSU = {}s'.format(tfin-tdebut)

tdebut = time.time()
contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
tfin = time.time()
print 'Temps en CV2 FIND CONTOURS = {}s'.format(tfin-tdebut)
    
print contours
cv2.drawContours(img, contours,-1, (0,0,255), 10)


cv2.imshow("image",img)
cv2.imshow("image2",img2)
cv2.imshow("image3",img3)
cv2.imshow("image4",img4)
cv2.waitKey(0)
cv2.destroyAllWindows()


